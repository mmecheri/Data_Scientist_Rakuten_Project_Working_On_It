import os

def list_directory_contents(path, indent_level=0, limit_images=3):
    """Recursively list all files and folders in a directory.
    
    - If a folder contains many images, display only a limited number (default: 3) and add '...'.
    - If a folder like `ipynb_checkpoints/` is not empty, display `[Non-empty folder]` instead of listing files.
    """
    try:
        items = sorted(os.listdir(path))  # Sort items for a clean display
        file_count = 0  # Count displayed files
        
        for item in items:
            item_path = os.path.join(path, item)
            
            # 📂 If it's a directory
            if os.path.isdir(item_path):
                # Special case: Handle `ipynb_checkpoints/` differently
                if item == ".ipynb_checkpoints":
                    checkpoint_files = os.listdir(item_path)  # Get files inside
                    if checkpoint_files:  # If the folder is not empty
                        print("│   " * indent_level + f"📂 {item}/ [Non-empty folder]")
                    else:
                        print("│   " * indent_level + f"📂 {item}/ [Empty]")
                
                else:
                    print("│   " * indent_level + f"📂 {item}/")  
                    
                    # If it's "image_test" or "image_train", apply limited display rule
                    if item in ["image_test", "image_train"]:
                        list_limited_images(item_path, indent_level + 1, limit_images)
                    else:
                        list_directory_contents(item_path, indent_level + 1, limit_images)
            
            # 📄 If it's a file (excluding images)
            elif not item.lower().endswith(('.jpg', '.jpeg', '.png')):  
                print("│   " * indent_level + f"📄 {item}")  
                file_count += 1
                
    except PermissionError:
        print("│   " * indent_level + "⚠️ [ACCESS DENIED]")

def list_limited_images(folder_path, indent_level, limit_images):
    """List a limited number of images from a folder, then display '...' if there are more."""
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for img in images[:limit_images]:  # Display only the first `limit_images` files
        print("│   " * indent_level + f"🖼️ {img}")
    
    if len(images) > limit_images:
        print("│   " * indent_level + "...")  # Indicate that there are more files

# 📌 Set the root directory
root_dir = os.getcwd()  # Change if needed (e.g., root_dir = "../data/")
print(f"📂 Listing contents of: {root_dir}\n")

# 📌 Run the script
list_directory_contents(root_dir)
