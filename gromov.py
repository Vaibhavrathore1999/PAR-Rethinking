import networkx as nx
import random
import itertools
import math

def calculate_gromov_hyperbolicity_approximation(graph, num_quadruples=1000):
    """
    Approximates the Gromov hyperbolicity (delta) of a graph.

    This function uses a sampling-based approach by checking the 4-point condition
    for a random subset of quadruples of nodes. It's an approximation
    and may not yield the exact delta, especially for small num_quadruples or
    highly structured graphs where specific quadruples might be missed.

    Args:
        graph (nx.Graph): A NetworkX graph where edge weights represent distances.
                         If no weights, default distance is 1 for connected nodes.
        num_quadruples (int): The number of random quadruples of nodes to test.
                              A larger number gives a better approximation but
                              increases computation time.

    Returns:
        float: The approximated Gromov delta. Returns 0.0 if the graph has
               fewer than 4 nodes or if no quadruples could be formed.
    """
    nodes = list(graph.nodes())
    if len(nodes) < 4:
        print("Graph has less than 4 nodes, cannot calculate hyperbolicity.")
        return 0.0

    # Pre-calculate all-pairs shortest paths
    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    def get_distance(u, v):
        """Helper to get distance between two nodes."""
        if u == v:
            return 0
        try:
            return all_shortest_paths[u][v]
        except KeyError:
            # If nodes are not connected, distance is effectively infinite
            return float('inf')

    max_delta = 0.0

    # Determine if we should sample or check all combinations
    total_possible_quadruples = math.comb(len(nodes), 4)
    
    if num_quadruples >= total_possible_quadruples:
        quadruples_to_check = itertools.combinations(nodes, 4)
        # print(f"Checking all {total_possible_quadruples} quadruples.")
    else:
        # Sample random quadruples
        # Create a list of all nodes and sample from it.
        # This can be slow if num_quadruples is very large but less than total_possible_quadruples
        # For efficiency, we generate unique random quadruples.
        
        # A more efficient way to get unique random samples without iterating combinations
        # is to generate indices or sample directly if the total_possible_quadruples is huge.
        # For now, let's stick to generating actual samples.
        sampled_quads = set()
        while len(sampled_quads) < num_quadruples:
            sample = tuple(sorted(random.sample(nodes, 4))) # Ensure unique and ordered for set
            sampled_quads.add(sample)
        quadruples_to_check = list(sampled_quads)
        # print(f"Checking {len(quadruples_to_check)} sampled quadruples.")


    for x, y, z, w in quadruples_to_check:
        d_xy = get_distance(x, y)
        d_zw = get_distance(z, w)
        d_xz = get_distance(x, z)
        d_yw = get_distance(y, w)
        d_xw = get_distance(x, w)
        d_yz = get_distance(y, z)

        # Skip if any pair is disconnected, as it skews the metric
        if any(d == float('inf') for d in [d_xy, d_zw, d_xz, d_yw, d_xw, d_yz]):
            continue

        # Calculate the three sums for the 4-point condition
        s1 = d_xy + d_zw
        s2 = d_xz + d_yw
        s3 = d_xw + d_yz

        sums = sorted([s1, s2, s3])

        current_delta = (sums[2] - sums[1]) / 2
        max_delta = max(max_delta, current_delta)

    return max_delta

# --- Attribute List and Conceptual Nodes ---
attributes = [
    'AGE_<16', 'AGE_16-25', 'AGE_26-35', 'AGE_36-45', 'AGE_46-60', 'AGE_>60',
    'UB_jacket', 'UB_sweater', 'UB_shirt', 'UB_tshirt', 'UB_suitmen',
    'UB_other', 'UB_saree', 'UB_kurta', 'UB_suitwomen', 'UB_stripes', 'UB_burqa',
    'LB_jeans', 'LB_shorts', 'LB_trousers', 'LB_saree', 'LB_salwar',
    'LB_leggings', 'LB_other', 'LB_burqa', 'LB_dhoti',
    'ACC_muffler', 'ACC_sunglasses', 'ACC_spectacles', 'ACC_other', 'ACC_beard', 'ACC_mask',
    'CARRY_backpack', 'CARRY_handbag', 'CARRY_suitcase', 'CARRY_other', 'CARRY_child',
    'FOOT_shoes', 'FOOT_sandals', 'FOOT_sneaker', 'FOOT_slippers',
    'GENDER_male', 'GENDER_female',
    'UBCOLOR_red', 'UBCOLOR_orange', 'UBCOLOR_blue', 'UBCOLOR_green', 'UBCOLOR_yellow',
    'UBCOLOR_black', 'UBCOLOR_white', 'UBCOLOR_mix', 'UBCOLOR_other', 'UBCOLOR_brown',
    'UBCOLOR_grey', 'UBCOLOR_pink', 'UBCOLOR_purple',
    'LBCOLOR_red', 'LBCOLOR_orange', 'LBCOLOR_blue', 'LBCOLOR_green', 'LBCOLOR_yellow',
    'LBCOLOR_black', 'LBCOLOR_white', 'LBCOLOR_mix', 'LBCOLOR_other', 'LBCOLOR_brown',
    'LBCOLOR_grey', 'LBCOLOR_pink', 'LBCOLOR_purple',
    'HAIR_black', 'HAIR_white', 'HAIR_other',
    'POSE_standing', 'POSE_sitting', 'POSE_lying',
    'SLEEVES_none', 'SLEEVES_long', 'SLEEVES_short',
    'VIEW_front', 'VIEW_back', 'VIEW_side',
    'HACC_Headgear', 'HACC_Helmet', 'HACC_WoolenCap', 'HACC_Hat'
]
conceptual_nodes = ['Person', 'Age', 'Gender', 'Upper Body (UB)', 'UB Color', 'Sleeves',
                    'Lower Body (LB)', 'LB Color', 'Accessories (ACC)', 'Carrying (CARRY)',
                    'Footwear (FOOT)', 'Hair', 'Pose', 'View', 'Head Accessories (HACC)']

# --- Function to create base tree graph ---
def create_base_tree_graph():
    graph = nx.Graph()
    graph.add_nodes_from(attributes + conceptual_nodes)

    graph.add_edge('Person', 'Age')
    graph.add_edges_from([('Age', attr) for attr in ['AGE_<16', 'AGE_16-25', 'AGE_26-35', 'AGE_36-45', 'AGE_46-60', 'AGE_>60']])

    graph.add_edge('Person', 'Gender')
    graph.add_edges_from([('Gender', attr) for attr in ['GENDER_male', 'GENDER_female']])

    graph.add_edge('Person', 'Upper Body (UB)')
    graph.add_edges_from([('Upper Body (UB)', attr) for attr in ['UB_jacket', 'UB_sweater', 'UB_shirt', 'UB_tshirt', 'UB_suitmen', 'UB_other', 'UB_saree', 'UB_kurta', 'UB_suitwomen', 'UB_stripes', 'UB_burqa']])
    graph.add_edge('Upper Body (UB)', 'UB Color')
    graph.add_edges_from([('UB Color', attr) for attr in ['UBCOLOR_red', 'UBCOLOR_orange', 'UBCOLOR_blue', 'UBCOLOR_green', 'UBCOLOR_yellow', 'UBCOLOR_black', 'UBCOLOR_white', 'UBCOLOR_mix', 'UBCOLOR_other', 'UBCOLOR_brown', 'UBCOLOR_grey', 'UBCOLOR_pink', 'UBCOLOR_purple']])
    graph.add_edge('Upper Body (UB)', 'Sleeves')
    graph.add_edges_from([('Sleeves', attr) for attr in ['SLEEVES_none', 'SLEEVES_long', 'SLEEVES_short']])

    graph.add_edge('Person', 'Lower Body (LB)')
    graph.add_edges_from([('Lower Body (LB)', attr) for attr in ['LB_jeans', 'LB_shorts', 'LB_trousers', 'LB_saree', 'LB_salwar', 'LB_leggings', 'LB_other', 'LB_burqa', 'LB_dhoti']])
    graph.add_edge('Lower Body (LB)', 'LB Color')
    graph.add_edges_from([('LB Color', attr) for attr in ['LBCOLOR_red', 'LBCOLOR_orange', 'LBCOLOR_blue', 'LBCOLOR_green', 'LBCOLOR_yellow', 'LBCOLOR_black', 'LBCOLOR_white', 'LBCOLOR_mix', 'LBCOLOR_other', 'LBCOLOR_brown', 'LBCOLOR_grey', 'LBCOLOR_pink', 'LBCOLOR_purple']])

    graph.add_edge('Person', 'Accessories (ACC)')
    graph.add_edges_from([('Accessories (ACC)', attr) for attr in ['ACC_muffler', 'ACC_sunglasses', 'ACC_spectacles', 'ACC_other', 'ACC_beard', 'ACC_mask']])

    graph.add_edge('Person', 'Carrying (CARRY)')
    graph.add_edges_from([('Carrying (CARRY)', attr) for attr in ['CARRY_backpack', 'CARRY_handbag', 'CARRY_suitcase', 'CARRY_other', 'CARRY_child']])

    graph.add_edge('Person', 'Footwear (FOOT)')
    graph.add_edges_from([('Footwear (FOOT)', attr) for attr in ['FOOT_shoes', 'FOOT_sandals', 'FOOT_sneaker', 'FOOT_slippers']])

    graph.add_edge('Person', 'Hair')
    graph.add_edges_from([('Hair', attr) for attr in ['HAIR_black', 'HAIR_white', 'HAIR_other']])

    graph.add_edge('Person', 'Pose')
    graph.add_edges_from([('Pose', attr) for attr in ['POSE_standing', 'POSE_sitting', 'POSE_lying']])

    graph.add_edge('Person', 'View')
    graph.add_edges_from([('View', attr) for attr in ['VIEW_front', 'VIEW_back', 'VIEW_side']])

    graph.add_edge('Person', 'Head Accessories (HACC)')
    graph.add_edges_from([('HACC_Headgear', attr) for attr in ['HACC_Headgear', 'HACC_Helmet', 'HACC_WoolenCap', 'HACC_Hat']]) # This was a typo, should be parent 'Head Accessories (HACC)'
    graph.add_edges_from([('Head Accessories (HACC)', attr) for attr in ['HACC_Headgear', 'HACC_Helmet', 'HACC_WoolenCap', 'HACC_Hat']])

    return graph

# --- Example Usage ---

# 1. Pure Tree Hierarchy (Expected Delta = 0.0)
print("--- Pure Tree Hierarchy ---")
tree_graph = create_base_tree_graph()

print(f"Nodes in pure tree graph: {len(tree_graph.nodes())}")
print(f"Edges in pure tree graph: {len(tree_graph.edges())}")

pure_tree_delta = calculate_gromov_hyperbolicity_approximation(tree_graph, num_quadruples=5000)
print(f"Approximated Gromov Delta (Pure Tree): {pure_tree_delta:.2f}")


# 2. Graph with Moderate Co-occurrence Edges (Expected Delta > 0)
print("\n--- Graph with Moderate Co-occurrence Edges ---")
co_occurrence_graph_moderate = tree_graph.copy() # Start with the tree structure

# Moderate set of plausible co-occurrence edges
moderate_co_occurrence_edges = [
    ('HACC_Hat', 'GENDER_male'),
    ('CARRY_backpack', 'AGE_<16'),
    ('UBCOLOR_red', 'LBCOLOR_red'),
    ('UB_saree', 'GENDER_female'),
    ('UB_burqa', 'LB_burqa'),
    ('SLEEVES_long', 'UB_sweater'),
    ('POSE_sitting', 'FOOT_slippers'),
    ('ACC_beard', 'GENDER_male'),
    ('ACC_mask', 'VIEW_front'),
    ('HAIR_white', 'AGE_>60'),
    ('UB_suitmen', 'GENDER_male'),
    ('UB_suitwomen', 'GENDER_female'),
    ('LB_jeans', 'AGE_16-25'),
    ('FOOT_sneaker', 'AGE_16-25'),
    ('CARRY_suitcase', 'POSE_standing'),
    ('UB_tshirt', 'SLEEVES_short'),
    ('LB_shorts', 'SLEEVES_short'),
    ('AGE_>60', 'ACC_spectacles'),
    ('HAIR_black', 'AGE_26-35'),
    ('VIEW_front', 'POSE_standing'),
]

co_occurrence_graph_moderate.add_edges_from(moderate_co_occurrence_edges)

print(f"Nodes in moderate co-occurrence graph: {len(co_occurrence_graph_moderate.nodes())}")
print(f"Edges in moderate co-occurrence graph: {len(co_occurrence_graph_moderate.edges())}")

co_occurrence_delta_moderate = calculate_gromov_hyperbolicity_approximation(co_occurrence_graph_moderate, num_quadruples=10000)
print(f"Approximated Gromov Delta (Moderate Co-occurrence): {co_occurrence_delta_moderate:.2f}")

# 3. Graph with More Extensive Co-occurrence Edges (Expected Delta > Moderate Delta)
print("\n--- Graph with Extensive Co-occurrence Edges ---")
co_occurrence_graph_extensive = tree_graph.copy() # Start with the tree structure

# Extensive set of plausible co-occurrence edges
extensive_co_occurrence_edges = [
    # General cultural/fashion links
    ('HACC_Hat', 'GENDER_male'),
    ('CARRY_backpack', 'AGE_<16'),
    ('UBCOLOR_red', 'LBCOLOR_red'), # Coordinated outfits
    ('UB_saree', 'GENDER_female'),
    ('UB_burqa', 'LB_burqa'), # Full traditional wear
    ('SLEEVES_long', 'UB_sweater'),
    ('POSE_sitting', 'FOOT_slippers'),
    ('ACC_beard', 'GENDER_male'),
    ('ACC_mask', 'VIEW_front'), # Often seen clearly from front
    ('HAIR_white', 'AGE_>60'),
    ('UB_suitmen', 'GENDER_male'),
    ('UB_suitwomen', 'GENDER_female'),
    ('LB_jeans', 'AGE_16-25'),
    ('FOOT_sneaker', 'AGE_16-25'),
    ('CARRY_suitcase', 'POSE_standing'),
    ('UB_tshirt', 'SLEEVES_short'),
    ('LB_shorts', 'SLEEVES_short'),
    ('AGE_>60', 'ACC_spectacles'),
    ('HAIR_black', 'AGE_26-35'),
    ('VIEW_front', 'POSE_standing'),

    # Added extensive edges
    
    ('UB_shirt', 'GENDER_male'),
    ('UB_kurta', 'GENDER_male'),
    ('UB_kurta', 'GENDER_female'), # Both genders wear kurta
    ('LB_salwar', 'GENDER_female'),
    ('LB_dhoti', 'GENDER_male'),
    ('ACC_sunglasses', 'VIEW_front'), # Typically worn to face forward
    ('ACC_sunglasses', 'UB_tshirt'), # Casual wear
    ('CARRY_handbag', 'GENDER_female'),
    ('FOOT_shoes', 'GENDER_male'),
    ('FOOT_sandals', 'GENDER_female'),
    ('HACC_Helmet', 'GENDER_male'),
    ('UBCOLOR_black', 'HAIR_black'), # Common combination
    ('LBCOLOR_black', 'UBCOLOR_white'), # Classic combination
    ('POSE_lying', 'AGE_<16'), # Children often lie down
    ('POSE_lying', 'AGE_>60'), # Older people resting
    ('SLEEVES_none', 'UB_tshirt'), # Sleeveless t-shirt
    ('UB_jacket', 'SLEEVES_long'), # Jackets usually have long sleeves
    ('LB_trousers', 'FOOT_shoes'), # Formal or semi-formal
    ('AGE_36-45', 'CARRY_child'), # Parents
    ('AGE_26-35', 'ACC_beard'), # Trendy beard age
    ('HACC_Headgear', 'GENDER_male'), # For religious or cultural reasons
    ('HACC_WoolenCap', 'AGE_<16'), # Keeping warm
    ('UBCOLOR_blue', 'LB_jeans'), # Blue jeans and blue top
    ('UBCOLOR_green', 'UB_shirt'), # Green shirt
    ('LBCOLOR_white', 'LB_trousers'), # White trousers
    ('VIEW_side', 'POSE_standing'), # Profile view
    ('UB_stripes', 'UB_tshirt'), # Striped t-shirt
]

co_occurrence_graph_extensive.add_edges_from(extensive_co_occurrence_edges)

print(f"Nodes in extensive co-occurrence graph: {len(co_occurrence_graph_extensive.nodes())}")
print(f"Edges in extensive co-occurrence graph: {len(co_occurrence_graph_extensive.edges())}")

co_occurrence_delta_extensive = calculate_gromov_hyperbolicity_approximation(co_occurrence_graph_extensive, num_quadruples=10000)
print(f"Approximated Gromov Delta (Extensive Co-occurrence): {co_occurrence_delta_extensive:.2f}")

# Verification and Comparison
print("\n--- Comparison Summary ---")
print(f"Pure Tree Delta: {pure_tree_delta:.2f}")
print(f"Moderate Co-occurrence Graph Delta: {co_occurrence_delta_moderate:.2f}")
print(f"Extensive Co-occurrence Graph Delta: {co_occurrence_delta_extensive:.2f}")

print("\n--- Verification ---")
if pure_tree_delta <= 0.1: # Allow for minor floating point inaccuracies
    print("✓ Pure tree delta is close to 0.0, as expected for a tree.")
else:
    print(f"✗ Pure tree delta ({pure_tree_delta:.2f}) is higher than expected.")

if co_occurrence_delta_moderate > pure_tree_delta:
    print(f"✓ Moderate co-occurrence delta ({co_occurrence_delta_moderate:.2f}) is higher than pure tree delta, as expected.")
else:
    print(f"✗ Moderate co-occurrence delta ({co_occurrence_delta_moderate:.2f}) is NOT higher than pure tree delta.")

if co_occurrence_delta_extensive > co_occurrence_delta_moderate:
    print(f"✓ Extensive co-occurrence delta ({co_occurrence_delta_extensive:.2f}) is higher than moderate co-occurrence delta, as expected.")
else:
    print(f"✗ Extensive co-occurrence delta ({co_occurrence_delta_extensive:.2f}) is NOT higher than moderate co-occurrence delta.")

if pure_tree_delta < co_occurrence_delta_moderate < co_occurrence_delta_extensive:
    print("\n✓ Overall trend: Delta increases with more meaningful co-occurrence edges, demonstrating increasing non-hyperbolicity.")
else:
    print("\n⚠ The expected increasing trend of delta was not observed. Review graph structures or sample size.")