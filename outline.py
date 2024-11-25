import os
import streamlit as st
import json
from collections import OrderedDict

# Path for outline storage
outline_file = "book_outline.json"


# Load outline from file
def load_outline():
    if os.path.exists(outline_file):
        with open(outline_file, "r") as file:
            return json.load(file, object_pairs_hook=OrderedDict)
    return OrderedDict()


# Save outline to file
def save_outline(outline):
    with open(outline_file, "w") as file:
        json.dump(outline, file, indent=4)


# Generate a unique key for each section or subsection
def generate_unique_key(outline, parent_key=None):
    if parent_key is None:
        return f"Section {len(outline) + 1}"
    else:
        subsections = outline[parent_key]["subsections"]
        return f"{parent_key}.Subsection {len(subsections) + 1}"


# Add a new section or subsection
def add_section(outline, parent_key=None, new_title="New Section"):
    if parent_key is None:
        # Add a main section
        key = generate_unique_key(outline)
        outline[key] = {"title": new_title, "subsections": OrderedDict()}
    else:
        # Add a subsection to the specified parent
        key = generate_unique_key(outline[parent_key]["subsections"], parent_key)
        outline[parent_key]["subsections"][key] = {"title": new_title, "subsections": OrderedDict()}
    save_outline(outline)


# Delete a section or subsection
def delete_section(outline, key_to_delete):
    if key_to_delete in outline:
        del outline[key_to_delete]
        save_outline(outline)
        return True
    for key, value in outline.items():
        if delete_section(value["subsections"], key_to_delete):
            save_outline(outline)
            return True
    return False


# Render the outline
def render_outline(outline, level=0):
    for key, value in outline.items():
        indent = "    " * level  # Indentation for hierarchical structure
        with st.container():
            st.markdown(f"**{indent}{value['title']} ({key})**")

            # Edit title
            new_title = st.text_input(f"Edit Title ({key})", value["title"], key=f"title_{key}")
            if new_title != value["title"]:
                value["title"] = new_title
                save_outline(outline)

            # Add subsection
            if st.button(f"+ Add Subsection to {key}", key=f"add_sub_{key}"):
                add_section(outline, parent_key=key, new_title="New Subsection")

            # Delete section
            if st.button(f"- Delete {key}", key=f"del_{key}"):
                delete_section(outline, key)
                st.experimental_rerun()

            # Render subsections
            if value["subsections"]:
                render_outline(value["subsections"], level=level + 1)


# Main app
def main():
    st.sidebar.header("Book Outline Editor")

    # Load the outline
    outline = load_outline()

    # Add a main section
    if st.sidebar.button("+ Add Main Section"):
        add_section(outline, new_title="New Main Section")

    # Display the outline
    st.header("Editable Book Outline")
    render_outline(outline)


if __name__ == "__main__":
    main()
