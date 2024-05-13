import os
def main() -> object:
    path=" "
    load_data(path)


def load_data(root_folder, file_suffix='_phy_k_manual'):
    """
    List all folder names under the specified root folder, including their absolute paths,
    and files that end with a specific suffix.

    Args:
    - root_folder (str): The path to the root folder from which to list subfolders and specific files.
    - file_suffix (str): The suffix that the files must end with to be included in the list.

    Returns:
    - dict: A dictionary with two keys: 'folders' containing paths to subfolders,
            and 'files' containing paths to files that end with the specified suffix.
    """

    file_paths = []

    # Walk through the directory
    folder_paths = [
        os.path.join(root_folder, name)
        for name in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, name))
    ]

        # Add files that end with the specified suffix
    for name in folder_paths:
        if name.endswith(file_suffix):
            file_paths.append(os.path.join(root_folder, name))

    print(file_paths)

    return file_paths


if __name__== "__main__":
    main()