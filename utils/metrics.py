import numpy as np
import torch
import cv2
from scipy.spatial import KDTree
from torch import Tensor


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
    ignore_index: int = None,
):
    """Average of Dice coefficient for all batches, or for a single mask"""

    input, target = input.flatten(0, 1), target.flatten(0, 1)

    if ignore_index is not None:
        mask = target != ignore_index
        input = input * mask
        target = target * mask

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def weighted_dice_score(
    input: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor = None,
    reduce_batch_first: bool = True,
    epsilon: float = 1e-10,
):
    """Average of Dice coefficient for all batches, or for a single mask"""

    # Ensure input and target are the same size
    assert input.size() == target.size()
    assert input.dim() == 4 or not reduce_batch_first  # Adjusted dimension check

    # Determine which dimensions to sum over
    sum_dim = (2, 3)  # Sum over height and width dimensions

    # Compute intersection and set sums class-wise
    inter = (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Compute Dice coefficient class-wise
    dice = (2 * inter + epsilon) / (sets_sum + epsilon)

    # Reduce across batches if required
    if reduce_batch_first:
        dice = dice.mean(dim=0)

    # Apply class weights, if provided
    if class_weights is not None:
        assert class_weights.size(0) == dice.size(-1)  # Adjusted dimension check
        dice = dice * class_weights

    return dice  # dice.mean()?


def weighted_dice_score2(output, target, device, num_classes=3):
    weights = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32, device=device)
    smooth = 1.0

    # Ensure the tensor dimensions are correct
    if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[1:]:
        raise ValueError("Shape mismatch between predicted and target")

    # Ensure the output is softmax
    output_softmax = torch.softmax(output, dim=1)

    # Initialize a tensor to accumulate weighted dice scores
    weighted_dice_score_accum = torch.tensor(0.0, dtype=torch.float32, device=device)

    # Calculate weighted Dice score
    for c in range(num_classes):
        pred_c = output_softmax[
            :, c, ...
        ]  # Probabilities that each pixel belongs to class c
        target_c = (
            target == c
        ).float()  # True (1) for pixels that actually belong to class c

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        weighted_dice_score_accum += weights[c] * dice_score

    # If you wish to update this tensor using backward pass, you may add requires_grad
    # weighted_dice_score_accum.requires_grad = True

    return weighted_dice_score_accum


def get_edge_coordinates(image: np.ndarray, class_label: int) -> np.ndarray:
    """Convert a single-class portion of an image to the coordinates of its edge pixels.

    Parameters:
    image (np.ndarray): Input image.
    class_label (int): The class label.

    Returns:
    np.ndarray: Coordinates of edge pixels.
    """
    class_img = (image == class_label).astype(np.uint8) * 255
    edges = cv2.Canny(class_img, 100, 200)
    return np.column_stack(np.where(edges > 0))


def mean_euclidean_distance_per_class(image1: np.ndarray, image2: np.ndarray) -> dict:
    """Calculate the mean Euclidean distance between edges for each class in two images.

    Parameters:
    image1 (np.ndarray): The first image.
    image2 (np.ndarray): The second image.

    Returns:
    dict: A dictionary mapping class labels to their mean Euclidean distances.
    """
    unique_classes = np.unique(np.concatenate([image1, image2]))
    mean_distances = {}

    for u in unique_classes:
        segmentation_edge = get_edge_coordinates(image1, u)
        gold_standard_edge = get_edge_coordinates(image2, u)

        N = len(segmentation_edge)
        if N == 0:
            mean_distances[u] = 0.0
            continue

        tree = KDTree(gold_standard_edge)
        distances, _ = tree.query(segmentation_edge, k=1)
        mean_distances[u] = float(np.mean(distances))

    return mean_distances


def final_score(preds, target):
    num_classes = 4  # number of classes

    dice_scores = dice_score_per_class(preds, target, num_classes)

    # Initialize scores dictionary
    scores = {}

    for tissue in TissueType:
        dice_value = dice_scores[tissue.value]

        # Slice out the predictions for the specific class
        sliced_preds = preds[:, tissue.value, :, :]

        MED_value = mean_euclidean_distance(sliced_preds, target)
        print(f"Class {tissue.name} MED: {MED_value}")

        # Calculate the weighted score for this tissue
        scores[tissue] = 0.5 * (10 * dice_value) + 0.5 * (MED_value + 1 - 0.3)

    # Calculate final score based on the weighted scores for each tissue type
    score = (
        0.4 * scores[TissueType.RNFL]
        + 0.3 * scores[TissueType.GCIPL]
        + 0.3 * scores[TissueType.CHOROID]
    )
    return score, dice_scores


# def differentiable_med(segmentation_edge, gold_standard_edge):
#     N = segmentation_edge.shape[0]
#     if N == 0:
#         return torch.tensor(0.0)

#     # Reshape the tensors to prepare for broadcasting
#     segmentation_edge = segmentation_edge.unsqueeze(1)  # shape (N, 1, 2)
#     gold_standard_edge = gold_standard_edge.unsqueeze(0)  # shape (1, M, 2)

#     # Compute pairwise squared distances
#     norm_s = torch.sum(segmentation_edge**2, dim=2, keepdim=True)  # shape (N, 1, 1)
#     norm_g = torch.sum(gold_standard_edge**2, dim=2, keepdim=True).permute(
#         0, 2, 1
#     )  # shape (1, 1, M)
#     dists = (
#         norm_s
#         + norm_g
#         - 2 * torch.matmul(segmentation_edge, gold_standard_edge.permute(0, 2, 1))
#     )  # shape (N, 1, M)

#     # Take sqrt to get Euclidean distances and then find the minimum along the last dimension
#     min_dists = torch.sqrt(torch.min(dists, dim=2)[0])  # shape (N,)

#     # Average over all points
#     mean_distance = torch.mean(min_dists)

#     return mean_distance


if __name__ == "__main__":
    # test mean euclidian distance on image data/train/Layer_show/0001.png to see whether it returns 0

    # load image
    image = cv2.imread("data/train/Layer_show/0001.png", cv2.IMREAD_GRAYSCALE)
    # find edge
    edge = find_edge(image)
    # calculate MED
    MED = mean_euclidean_distance(edge, edge)
    print(MED)
    edge_tensor = torch.tensor(edge, dtype=torch.float32, requires_grad=True)

    # also test it on two different images
    image2 = cv2.imread("data/train/Layer_show/0002.png", cv2.IMREAD_GRAYSCALE)
    edge2 = find_edge(image2)
    edge2_tensor = torch.tensor(edge2, dtype=torch.float32, requires_grad=True)
    MED2 = mean_euclidean_distance(edge, edge2)
    print(MED2)
    MED3 = differentiable_med(edge_tensor, edge2_tensor)
    print(MED3.item())
    MED3.backward()