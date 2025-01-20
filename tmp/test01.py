import maya.cmds as cmds

def rename_and_copy_meshes_to_new_group(source_group, new_name_prefix, target_group):
    """
    Rename all meshes under a specified group, delete non-mesh nodes, 
    and copy the renamed meshes to a new target group.

    Args:
        source_group (str): The name of the source group (empty object) to search under.
        new_name_prefix (str): The prefix for the new names of the meshes.
        target_group (str): The name of the target group to create and place the renamed meshes under.
    """
    if not cmds.objExists(source_group):
        cmds.error(f"Source group '{source_group}' does not exist.")
        return

    # Ensure the target group exists, or create it
    if not cmds.objExists(target_group):
        cmds.group(empty=True, name=target_group)

    # Get all descendants of the source group
    descendants = cmds.listRelatives(source_group, allDescendents=True, fullPath=True) or []

    mesh_count = 0

    for node in descendants:
        # Check if the node is a mesh
        if cmds.nodeType(node) == "mesh":
            # Get the transform node of the mesh
            transform_node = cmds.listRelatives(node, parent=True, fullPath=False)[0]
            
            # Rename the transform node
            new_name = f"{new_name_prefix}_{mesh_count+1}"
            renamed_node = cmds.rename(transform_node, new_name)
            
            # Duplicate the renamed mesh and parent it to the target group
            duplicated_node = cmds.duplicate(renamed_node, name=new_name)[0]
            cmds.parent(duplicated_node, target_group)

            mesh_count += 1
        else:
            # Delete non-mesh nodes
            cmds.delete(node)

    print(f"Copied and renamed {mesh_count} meshes from '{source_group}' to '{target_group}'.")

# Example usage:
# Replace 'source_group' with your source group name, 'mesh' with your desired prefix,
# and 'target_group' with the target group name
rename_and_copy_meshes_to_new_group("source_group", "mesh", "target_group")