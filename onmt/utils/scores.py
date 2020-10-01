import abc

from rouge import FilesRouge
from sacrebleu import corpus_bleu
from onmt.utils import ClassRegistry

class Scorer(abc.ABC):
    """Scores hypotheses against references."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """The scorer name."""
        return self._name

    @property
    def scores_name(self):
        """The names of returned scores."""
        return {self._name}

    @abc.abstractmethod
    def __call__(self, ref_path, hyp_path):
        """Scores hypotheses.

    Args:
      ref_path: Path to the reference.
      hyp_path: Path to the hypotheses.

    Returns:
      The score or dictionary of scores.
    """
        raise NotImplementedError()

    def lower_is_better(self):
        """Returns ``True`` if a lower score is better."""
        return False

    def higher_is_better(self):
        """Returns ``True`` if a higher score is better."""
        return not self.lower_is_better()


_SCORERS_REGISTRY = ClassRegistry(base_class=Scorer)
register_scorer = _SCORERS_REGISTRY.register  # pylint: disable=invalid-name

@register_scorer(name="rouge")
class ROUGEScorer(Scorer):
    """ROUGE scorer based on https://github.com/pltrdy/rouge."""

    def __init__(self):
        super(ROUGEScorer, self).__init__("rouge")

    @property
    def scores_name(self):
        return {"rouge-1", "rouge-2", "rouge-l"}

    def __call__(self, ref_path, hyp_path):
        scorer = FilesRouge(metrics=list(self.scores_name))
        rouge_scores = scorer.get_scores(hyp_path, ref_path, avg=True)
        return {name: rouge_scores[name]["f"] for name in self.scores_name}


@register_scorer(name="bleu")
class BLEUScorer(Scorer):
    """Scorer using sacreBLEU."""

    def __init__(self):
        super(BLEUScorer, self).__init__("bleu")

    def __call__(self, ref, hyp):
        bleu = corpus_bleu(ref, [hyp], force=True)
        return bleu.score


def make_ext_evaluators(names):
    if not isinstance(names, list):
        names = [names]
    scorers = []

    for name in names:
        scorer_class = _SCORERS_REGISTRY.get(name.lower())
        if scorer_class is None:
            raise ValueError("No scorer associated with the name: {}".format(name))
        scorers.append(scorer_class())
    return scorers
