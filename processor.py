import torch
import torchaudio.compliance.kaldi as kaldi


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    # this contains the abs length of the padding for each dimension.
    pads = []
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1
    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals

def batch_pad_right(tensors: list, mode="constant", value=0,val_index=-1):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")
    # tensors = list(map(list,tensors))

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        any(
            [tensors[i].ndim == tensors[0].ndim for i in range(
                1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != (tensors[0].ndim - 1):
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for last one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[val_index])

    batched = torch.stack(batched)
    
    return batched, torch.tensor(valid)



class InputSequenceNormalization(object):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.

    Example
    -------
    >>> import torch
    >>> norm = InputSequenceNormalization()
    >>> input = torch.randn([101, 20])
    >>> feature = norm(inputs)
    """


    def __init__(
        self,
        mean_norm=True,
        std_norm=False,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = 1e-10
    def __call__(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A tensor `[t,f]`.

        
        """
        if self.mean_norm:
            mean = torch.mean(x, dim=0).detach().data
        else:
            mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            std = torch.std(x, dim=0).detach().data
        else:
            std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        std = torch.max(
            std, self.eps * torch.ones_like(std)
        )
        x = (x - mean.data) / std.data


        return x

class rawWav2KaldiFeature(object):
    """ This class extract features as kaldi's compute-mfcc-feats.

    Arguments
    ---------
    feat_type: str (fbank or mfcc).
    feature_extraction_conf: dict
        The config according to the kaldi's feature config.
    """

    def __init__(self,feature_type='mfcc',kaldi_featset={},mean_var_conf={}):
        super().__init__()
        assert feature_type in ['mfcc','fbank']
        self.feat_type=feature_type
        self.kaldi_featset=kaldi_featset
        if self.feat_type=='mfcc':
            self.extract=kaldi.mfcc
        else:
            self.extract=kaldi.fbank
        if mean_var_conf is not None:
            self.mean_var=InputSequenceNormalization(**mean_var_conf)
        else:
            self.mean_var=torch.nn.Identity()

    def __call__(self,wav,wav_len,sample_rate=16000):
        """ make features.
            wav : B x T
            wav_len : B
        """


        self.kaldi_featset['sample_frequency'] = sample_rate
        lens = wav_len
        waveforms = wav * (1 << 15)
        feats = []
        labels=[]
        keys=[]
        feature_len=[]
        # lens=lens*waveforms.shape[-1]
        # print(lens)
        for i,waveform in enumerate(waveforms):
            # print(waveform.shape)
            if len(waveform.shape)==1:
                # add channel
                waveform=waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            waveform= waveform[:,:lens[i].long()]
            # print(wav.shape)
            # print(waveform.shape,print(lens[i]))
            feat=self.extract(waveform,**self.kaldi_featset)

            # if(torch.any((torch.isnan(feat)))):
            #     logging.warning('Failed to make featrue for {}, aug version:{}')
            #     continue
            feat=self.mean_var(feat)
            feat=feat.transpose(-1,-2)
            # print(feat.shape)
            feats.append(feat)
            feature_len.append(feat.shape[0])
            # labels.append(label[i])
            # keys.append(key)
        # feats=torch.cat([feat.unsqueeze(0) for feat in feats],dim=0)
        feats,_= batch_pad_right(feats)
        # max_len = max([feat.size(0) for feat in feats])
        return feats