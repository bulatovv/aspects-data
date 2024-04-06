from torch import Tensor
from typing import Optional, Callable

def make_embeds_sim_classifier(
    classes: list[str], 
    embeddings_model: Callable[[str], Tensor],
    sim_threshold: float,
    sim_metric: Callable[[Tensor, Tensor], float]
) -> Callable[[str], Optional[str]]:
    """
    Creates a classifier that classifies some text by relation to the one of aspects (classes).

    :param list[str] classes: Classes that can be contained in texts
    :param Callable[[str], torch.FloatTensor]: Embedings model text -> vactor
    :param float sim_threshold: Threshold of similarity. If no one crosses None returns
    :param Callable[[torch.FloatTensor, torch.FloatTensor], float] sim_metric: Similarity metric e.g. cosine similarity
    :return Callable text classifier
    :rtype: Callable[[str], Sentiment]
    """
    # Compute class embeddings once
    cls_embeddings = [(cls, embeddings_model(cls)) for cls in classes]

    # Define the classifier function
    def embeds_sim_classifier(text: str) -> Optional[str]:
        """
        Classifies given text by aspects.
        The class (of classes) with the highest similarity measure (sim_metric) is selected if this similarity is above the threshold (sim_threshold)

        :param str text: A text that may relate to one of the aspects (classes)
        :return: An aspect (class) or None
        :rtype: Optional[str]
        """
        text_embedding = embeddings_model(text)
        # Calculate similarities and get the best match
        sims = [(sim_metric(text_embedding, cls_embedding), cls) for cls, cls_embedding in cls_embeddings]
        best_sim, best_cls = max(sims, key=lambda pair: pair[0])
        # Return the best class if above the similarity threshold, else None
        return best_cls if best_sim > sim_threshold else None
    return embeds_sim_classifier
