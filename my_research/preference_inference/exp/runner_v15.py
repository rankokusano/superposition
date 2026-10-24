"""
v15 stage 1 (docs/v15_experiment_log.md Sec.8): the paper's runner with an extra colour term.

PredictionFeaturePredictionColourRunner = PredictionFeaturePredictionRunner (unchanged loss on the
predicted view) + colour_lambda * the same loss function on the chroma of the view
(chroma = each pixel's RGB minus its channel mean, on the [-1, 1] scaled view). The chroma error is large
where a coloured landmark is drawn in another colour or as the yellow average, and where a gray cube is
drawn coloured (or a colour drawn gray); it is ~0 where floor, ground, sky and gray stay unchanged.
colour_lambda comes from config.train.colour_lambda.
"""

from . import runner as runners


def chroma(v):
    """v: (B, 3, H, W) -> (B, 3, H, W)."""
    return v - v.mean(1, keepdim=True)


class PredictionFeaturePredictionColourRunner(runners.PredictionFeaturePredictionRunner):
    def calc_loss(self, model, loss, pred, truth, epoch, t, config):
        super().calc_loss(model, loss, pred, truth, epoch, t, config)
        loss_fn = getattr(runners.F, config.train.loss_function)
        loss["self_vision_colour"] += config.train.colour_lambda * loss_fn(
            chroma(pred["self_vision"]), chroma(truth["self_vision"]))
