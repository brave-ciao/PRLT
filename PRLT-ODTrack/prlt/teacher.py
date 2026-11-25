from contextlib import contextmanager
import torch


@contextmanager
def temporary_teacher_eval(net):
    modules = tuple(net.modules())
    modes = tuple(module.training for module in modules)
    query = net.track_query
    try:
        net.eval()
        with torch.no_grad():
            yield net
    finally:
        net.track_query = query
        for module, mode in zip(modules, modes):
            module.training = mode


def paired_teacher_forward(net, templates, current, reference):
    if not isinstance(templates, list) or not templates:
        raise ValueError("ODTrack teacher requires its original template list")
    if current.shape != reference.shape or any(z.shape[0] != current.shape[0] for z in templates):
        raise ValueError("Teacher template/current/reference batch shapes do not match")
    with temporary_teacher_eval(net):

        net.track_query = None
        now = net(template=templates, search=[current], ce_template_mask=None,
                  ce_keep_rate=None, return_last_attn=False)[0]
        net.track_query = None
        before = net(template=templates, search=[reference], ce_template_mask=None,
                     ce_keep_rate=None, return_last_attn=False)[0]
    return now, before
