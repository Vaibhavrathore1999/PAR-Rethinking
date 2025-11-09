import pickle
import os

pkl_file_path = '/users/student/pg/pg23/vaibhav.rathore/PAR/Rethinking_of_PAR/data/VRLChallenge_2/dataset_all.pkl' # Adjust this path if needed

if not os.path.exists(pkl_file_path):
    print(f"Error: Pickle file not found at {pkl_file_path}")
else:
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded {pkl_file_path}")
        print(f"Type of loaded data: {type(data)}")

        # Depending on the type, you can inspect it further
        if isinstance(data, dict):
            print(f"Keys in the dictionary: {data.keys()}")
            # Print first few items for some keys, or specific keys if you know them
            for key, value in data.items():
                print(f"\n--- Key: '{key}' ---")
                if isinstance(value, list) and len(value) > 0:
                    print(f"First 5 items (or less): {value[:5]}")
                    print(f"Total items: {len(value)}")
                elif isinstance(value, dict):
                    print(f"Sub-keys: {value.keys()}")
                    # Optionally print a nested value
                    # if 'train_imgid' in data and len(data['train_imgid']) > 0:
                    #     print(f"Example train_imgid: {data['train_imgid'][0]}")
                else:
                    print(f"Value: {value}")
        elif isinstance(data, list) and len(data) > 0:
            print(f"First 5 items (or less): {data[:5]}")
            print(f"Total items: {len(data)}")
        else:
            print(f"Data: {data}")
        print("\n--- Partition Sizes ---")
        if 'partition' in data:
            for split_name, split_data in data['partition'].items():
                # 'split_data' is likely a list or numpy array of indices
                print(f"Split '{split_name}': {len(split_data)} samples")
        else:
            print("Error: 'partition' key not found in dataset.")

        # Also, check the total number of images and labels
        if 'image_name' in data:
            print(f"\nTotal 'image_name' entries: {len(data['image_name'])}")
        if 'label' in data:
            print(f"Total 'label' entries: {len(data['label'])}")

    except Exception as e:
        print(f"Error loading or inspecting pickle file: {e}")