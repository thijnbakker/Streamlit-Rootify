import networkx as nx
import numpy as np
import cv2
import pandas as pd
import logging
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

logger = logging.getLogger(__name__)


def crop_top_roots(root_areas, labels):
    """
    Crop individual root regions from a labeled mask based on root area info.

    Args:
        root_areas (list of tuples): List of (root_id, area) tuples representing labeled root components.
        labels (numpy.ndarray): 2D array of the same size as the mask with integer labels for connected components.

    Returns:
        List[numpy.ndarray]: List of cropped root masks as uint8 binary images (0 or 255).
    """

    logger.info("Cropping top roots from the labeled mask...")

    cropped_roots = []
    for root_id, _ in root_areas:
        root = (labels == root_id).astype(np.uint8) * 255
        coords = np.argwhere(root)
        logger.debug(f"Root ID: {root_id}, Coordinates: {coords}")
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        logger.debug(f"Cropping coordinates: y_min={y_min}, x_min={x_min}, y_max={y_max}, x_max={x_max}")
        cropped_root = root[y_min:y_max + 1, x_min:x_max + 1]
        cropped_roots.append(cropped_root)
        logger.debug(f"Cropped root shape: {cropped_root.shape}")
    logger.info(f"Total cropped roots: {len(cropped_roots)}")
    return cropped_roots


def process_roots(root_areas, labels):
    """
    Process individual root masks by cropping and skeletonizing.

    Args:
        root_areas (list of tuples): List of (root_id, area) tuples for connected root regions.
        labels (numpy.ndarray): 2D labeled image of connected components.

    Returns:
        List[numpy.ndarray]: List of skeletonized root images (binary skeleton masks).
    """

    logger.info("Processing roots by cropping and skeletonizing...")
    processed_roots = []
    for root_id, _ in root_areas:
        root = (labels == root_id).astype(np.uint8) * 255
        coords = np.argwhere(root)
        logger.debug(f"Root ID: {root_id}, Coordinates: {coords}")
        if coords.size == 0:  # Skip empty regions
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped_root = root[y_min:y_max + 1, x_min:x_max + 1]
        logger.debug(f"Cropping coordinates: y_min={y_min}, x_min={x_min}, y_max={y_max}, x_max={x_max}")
        _, binary_image = cv2.threshold(cropped_root, 0, 255, cv2.THRESH_BINARY)

        # Check if the root is empty before proceeding
        if np.count_nonzero(binary_image) == 0:
            logger.warning(f"Root ID {root_id} is empty after thresholding. Skipping skeletonization.")
            continue

        skeleton = skeletonize(binary_image // 255)
        processed_roots.append(skeleton)
        logger.info("Skeletonization completed for root ID: {}".format(root_id))
    return processed_roots  # Returns list of skeletonized roots


def calculate_path_length(binary_image):
    """
    Calculate the shortest path length along the skeleton of a binary root image.

    Args:
        binary_image (numpy.ndarray): Binary mask of a root (non-zero pixels = root).

    Returns:
        float: Path length along the skeleton graph between the smallest and largest node IDs.
               Returns 0 if skeleton graph is empty or path does not exist or length < 20.
    """
    logger.info("Calculating path length from the binary image...")
    skeleton_image = skeletonize(binary_image)
    logger.debug(f"Skeleton image shape: {skeleton_image.shape}, non-zero pixels: {np.count_nonzero(skeleton_image)}")

    try:
        skeleton_object = Skeleton(skeleton_image)
    except ValueError as e:
        logger.error(f"Error creating Skeleton object: {e}")
        return 0

    branch_data = summarize(skeleton_object)
    logger.debug(f"Branch data:\n{branch_data}")

    G = nx.from_pandas_edgelist(branch_data, source='node-id-src', target='node-id-dst', edge_attr='branch-distance')

    if len(G.nodes) == 0:
        logger.warning("Graph is empty. No nodes to calculate a path.")
        return 0

    first_node = min(G.nodes)
    last_node = max(G.nodes)

    if nx.has_path(G, first_node, last_node):
        path_len = nx.dijkstra_path_length(G, first_node, last_node, weight='branch-distance')
    else:
        logger.warning(f"No path between {first_node} and {last_node}.")
        return 0

    if 0 < path_len < 20:
        logger.info(f"Path length {path_len} is less than 20, setting to 0.")
        path_len = 0

    logger.info(f"Calculated path length: {path_len}")
    logger.debug(f"Path length: {path_len}, First node: {first_node}, Last node: {last_node}")
    return path_len


def tresh_mask(predicted_mask):
    """
    Remove padding from a predicted mask, normalize it, and apply Otsu thresholding.

    Args:
        predicted_mask (numpy.ndarray): Mask predicted by model (possibly padded).
    Returns:
        numpy.ndarray: Thresholded binary mask (uint8, 0 or 255).
    """

    normalized_mask = cv2.normalize(predicted_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(normalized_mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug(f"Thresholded mask shape: {thresh.shape}, non-zero pixels: {np.count_nonzero(thresh)}")

    return thresh


def find_root_tips(binary_image):
    """
    Identify root tip coordinates by analyzing the skeleton endpoints of the largest skeletons.

    Args:
        binary_image (numpy.ndarray): Binary mask representing roots.

    Returns:
        pandas.DataFrame: DataFrame with info on top 5 largest skeleton endpoints sorted by x-coordinate.
        List[Tuple[int, int]]: List of (y, x) coordinates of root tips sorted by x-coordinate.
    """

    # Skeletonize the binary image
    skeleton_image = skeletonize(binary_image)

    # Convert the skeleton image to a Skeleton object
    try:
        skeleton_object = Skeleton(skeleton_image)
    except ValueError as e:
        logger.error(f"Error creating Skeleton object: {e}")
        return []

    # Summarize the skeleton and create a dataframe of branches
    branch_data = summarize(skeleton_object)

    tip_coordinates = []

    # Calculate the size of each skeleton by summing the branch distances
    skeleton_sizes = branch_data.groupby("skeleton-id")["branch-distance"].sum()

    # Select the top 5 biggest skeletons
    top_5_skeletons = skeleton_sizes.nlargest(5).index
    logger.debug(f"Top 5 skeletons: {top_5_skeletons}")

    # For each of the top 5 biggest skeletons, find the max node-id-dst
    max_node_dst_info = []

    for skeleton_id in top_5_skeletons:
        # Filter data for the current skeleton
        skeleton_df = branch_data[branch_data["skeleton-id"] == skeleton_id]

        # Find the row with the max node-id-dst for this skeleton
        max_node_dst_row = skeleton_df.loc[skeleton_df["node-id-dst"].idxmax()]

        # Extract the image-coord-dst-0 and image-coord-dst-1
        image_coord_dst_0 = max_node_dst_row["image-coord-dst-0"]
        image_coord_dst_1 = max_node_dst_row["image-coord-dst-1"]

        # Store the result
        max_node_dst_info.append({
            "skeleton-id": skeleton_id,
            "max-node-id-dst": max_node_dst_row["node-id-dst"],
            "image-coord-dst-0": image_coord_dst_0,
            "image-coord-dst-1": image_coord_dst_1
        })

        # Store the result in the list
        tip_coordinates.append((image_coord_dst_0, image_coord_dst_1))
        logger.debug(f"Tip coordinates for skeleton {skeleton_id}: {tip_coordinates[-1]}")

    # Sort the result by "image-coord-dst-1"
    max_node_dst_info = sorted(max_node_dst_info, key=lambda x: x["image-coord-dst-1"])

    # Sort the tip_coordinates by the second element (image_coord_dst_1)
    tip_coordinates = sorted(tip_coordinates, key=lambda x: x[1])

    # Convert to DataFrame for better readability
    max_node_dst_df = pd.DataFrame(max_node_dst_info)

    return max_node_dst_df, tip_coordinates


def find_root_lengths(binary_image):
    """
    Estimate root lengths by dividing the image into horizontal regions, extracting largest root components,
    skeletonizing them, and computing path lengths.

    Args:
        binary_image (numpy.ndarray): Binary mask representing roots.

    Returns:
        List[float]: List of estimated root lengths for each predefined horizontal region.
    """

    # Define regions for cropping (x-axis ranges)
    regions = [(10, 700), (600, 1200), (1100, 1700), (1500, 2300), (2100, 2700)]
    all_root_lengths = []  # List to store root lengths

    # Process each region
    for i, (start, end) in enumerate(regions):
        logger.debug(f"Processing region {i+1}: start={start}, end={end}")

        # Crop region from the binary image (skip top 300px to exclude noise)
        regioned_image = binary_image[300:, start:end]
        logger.debug(f"Region shape: {regioned_image.shape}, non-zero pixels: {np.count_nonzero(regioned_image)}")

        # Compute connected components
        _, labels = cv2.connectedComponents(regioned_image.astype(np.uint8))

        # Extract all root areas
        root_areas = [
            (root_id, np.sum(labels == root_id))
            for root_id in np.unique(labels) if root_id != 0
        ]

        if root_areas:
            # Select the largest root by area
            largest_root = max(root_areas, key=lambda x: x[1])
            logger.debug(f"Largest root ID: {largest_root[0]}, Area: {largest_root[1]}")

            # Crop the largest root
            cropped_roots = crop_top_roots([largest_root], labels)
            if cropped_roots:
                # Skeletonize the binary root mask
                skeleton = skeletonize(cropped_roots[0] > 0)

                # Compute the path length
                root_length = calculate_path_length(skeleton)
                logger.debug(f"Root length for region {i+1}: {root_length}")
            else:
                logger.warning(f"No cropped root found for region {i+1}.")
                root_length = 0
        else:
            logger.warning(f"No roots found in region {i+1}.")
            root_length = 0

        all_root_lengths.append(root_length)

    return all_root_lengths
