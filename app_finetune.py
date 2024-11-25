import os
import time
import streamlit as st
import json
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from embeddings import load_embeddings_offline, process_pdfs_in_folder_and_save_embeddings
from htmlTemplates import css, bot_template, user_template

# Page configuration
st.set_page_config(page_title="Book Writing with Progress", page_icon=":books:")

# Load environment variables from the .env file
load_dotenv()

# Retrieve necessary API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API keys
if not google_api_key or not google_cse_id:
    st.error("Google API key or CSE ID not found in environment variables.")
    st.stop()
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables.")
    st.stop()

# Paths for chat history and book progress files
history_file = os.path.join(os.getcwd(), "chat_history.json")
book_file = os.path.join(os.getcwd(), "book_progress.txt")

# Load chat history from file
def load_chat_history():
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            return json.load(file)
    return []

# Save chat history to file
def save_chat_history(chat_history):
    with open(history_file, 'w') as file:
        json.dump(chat_history, file)

# Initialize writing progress file for book
if not os.path.exists(book_file):
    with open(book_file, 'w') as f:
        f.write("Book Writing Progress:\n\n")

# Append content to the book progress file
def append_to_book(content):
    if content:
        with open(book_file, 'a') as f:
            f.write(f"{content}\n\n")

# Google Custom Search function to retrieve URLs only
# Google Custom Search function to retrieve URLs and extended content
def google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        query_with_domains = (
            f"{query} site:scholar.google.com OR site:ascelibrary.org OR site:icevirtuallibrary.com OR "
            f"site:engineeringvillage.com OR site:springer.com OR site:sciencedirect.com OR site:wiley.com OR "
            f"site:tandfonline.com OR site:ieeexplore.ieee.org OR site:researchgate.net OR site:mdpi.com OR "
            f"site:cambridge.org OR site:oup.com OR site:jstor.org OR site:arxiv.org OR site:frontiersin.org OR "
            f"site:wikipedia.org"
        )
        result = service.cse().list(q=query_with_domains, cx=google_cse_id, num=10).execute()  # Fetch up to 10 results
        search_items = result.get('items', [])
        
        links = [item.get('link', '') for item in search_items]
        combined_answer = " ".join([item.get('snippet', '') + " " + item.get('htmlSnippet', '') for item in search_items])
        return combined_answer, links
    except Exception as e:
        st.error(f"Error in Google Search: {e}")
        return "", []

# Adjusted function to generate detailed GPT responses
def handle_question_with_search(question, use_online_search=False):
    response = ""
    links = []
    
    if use_online_search:
        combined_answer, links = google_search(question)
        gpt_prompt = (
            f"Here is a comprehensive set of information gathered for the question: '{question}'.\n\n"
            f"Context:\n{combined_answer}\n\n"
            f"Using the context above, provide a detailed and comprehensive explanation. Include:\n"
            f"- Technical insights and methodologies\n"
            f"- Examples and use cases\n"
            f"- Additional perspectives or interpretations\n\n"
            f"Ensure the response is structured with clear headings, bullet points, and multiple paragraphs where necessary. "
            f"The output should be suitable for academic or professional purposes.\n\n"
            f"Question: {question}\nAnswer:"
        )
        
        llm = ChatOpenAI(temperature=0.7, max_tokens=4096, openai_api_key=openai_api_key)  # Increased max_tokens
        final_response = llm([HumanMessage(content=gpt_prompt)])
        response = final_response.content if final_response else "No response generated."
    else:
        response = handle_offline_retrieval(question, retrieve_multiple=True)
        if not response:
            response = "No relevant information found in the offline embeddings."
        
    if response:
        append_to_book(response)
        display_response(question, response, links)
    else:
        st.warning("No response generated. Try rephrasing your question.")

    # Append question and response to chat history
    st.session_state.chat_history.append({'role': 'user', 'content': question})
    st.session_state.chat_history.append({'role': 'bot', 'content': response})
    save_chat_history(st.session_state.chat_history)


# Function to handle questions with optional web search integration
# Retrieve multiple passages based on embeddings
def handle_offline_retrieval(question, retrieve_multiple=False):
    if st.session_state.conversation:
        if retrieve_multiple:
            responses = []
            retrieval_results = st.session_state.vectorstore.as_retriever().get_relevant_documents(question)
            for doc in retrieval_results:
                gpt_prompt = (
                    f"Given the following retrieved passage, write a detailed and well-structured explanation on '{question}', "
                    f"using sections, bullet points, and clear paragraphs as necessary:\n\n"
                    f"{doc.page_content}\n\nAnswer:"
                )
                llm = ChatOpenAI(temperature=0.7, max_tokens=3500, openai_api_key=openai_api_key)
                response = llm([HumanMessage(content=gpt_prompt)])
                if response:
                    responses.append(response.content)
            
            return "\n\n".join(responses) if responses else "No detailed response generated."
        else:
            response = st.session_state.conversation({'question': question, 'chat_history': []})
            return response['answer']
    else:
        st.write("Embeddings not loaded. Please check the FAISS index path.")
        return ""

# Display the response with proper HTML formatting
def display_response(question, response, links=None):
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    formatted_response = format_response(response)  # Formatting for better readability
    st.markdown(formatted_response, unsafe_allow_html=True)  # Display formatted response directly with markdown
    if links:
        st.write("\n\n**Citations:**")
        for link in links:
            st.write(f"- [{link}]({link})")

# Format response to replicate the sidebar structure with consistent headings and selective bullet points
# Format response to precisely replicate the sidebar structure with correct headings, bullet points, and indentation
# Format response with strict HTML formatting to prevent Markdown artifacts like ** and ###
# Format response to add sections, bullets, and better readability with black font
def format_response(response):
    """
    Dynamically formats the AI response into well-structured sections:
    1. Introduction
    2. Detailed Explanation (includes all key information)
    3. Conclusion
    Sections are inferred from the content and styled for better readability.
    """
    # Splitting the response into paragraphs
    paragraphs = response.split("\n\n")
    formatted_response = "<div style='color: black; font-size: 18px; line-height: 1.8em;'>"

    # Initialize containers for sections
    intro, explanation, conclusion = [], [], []
    current_section = "Introduction"

    # Dynamically assign content to sections based on semantic meaning
    for paragraph in paragraphs:
        # Heuristic checks for section categorization
        if "introduction" in paragraph.lower():
            current_section = "Introduction"
        elif "conclusion" in paragraph.lower() or paragraph.startswith("In conclusion"):
            current_section = "Conclusion"
        else:
            current_section = "Detailed Explanation"

        # Append paragraph to the correct section
        if current_section == "Introduction":
            intro.append(paragraph)
        elif current_section == "Detailed Explanation":
            explanation.append(paragraph)
        elif current_section == "Conclusion":
            conclusion.append(paragraph)

    # Add Introduction section
    if intro:
        formatted_response += "<h2 style='margin-top: 20px; color: black;'>Introduction</h2>"
        for para in intro:
            formatted_response += f"<p style='margin-bottom: 10px;'>{para}</p>"

    # Add Detailed Explanation section
    if explanation:
        formatted_response += "<h2 style='margin-top: 20px; color: black;'>Detailed Explanation</h2>"
        for para in explanation:
            # Add bullet points dynamically for key points
            if para.startswith("- "):
                formatted_response += f"<ul style='padding-left: 20px;'><li>{para[2:]}</li></ul>"
            else:
                formatted_response += f"<p style='margin-bottom: 10px;'>{para}</p>"

    # Add Conclusion section
    if conclusion:
        formatted_response += "<h2 style='margin-top: 20px; color: black;'>Conclusion</h2>"
        for para in conclusion:
            formatted_response += f"<p style='margin-bottom: 10px;'>{para}</p>"

    # Close the container
    formatted_response += "</div>"

    return formatted_response



# Process uploaded PDFs for embedding-based retrieval
def process_uploaded_pdfs(uploaded_pdfs):
    temp_dir = "uploaded_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    for uploaded_pdf in uploaded_pdfs:
        file_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
    process_pdfs_in_folder_and_save_embeddings(temp_dir, "faiss_index")

    try:
        st.session_state.vectorstore = load_embeddings_offline("faiss_index")
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        st.success("PDFs processed, embeddings updated, and FAISS index reloaded.")
    except Exception as e:
        st.error(f"Failed to reload embeddings after PDF processing: {e}")

# Create conversation chain for embeddings
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7, max_tokens=4000, openai_api_key=openai_api_key)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=None
    )
    return conversation_chain

# Display chat history in sidebar
def display_memory_in_sidebar():
    st.sidebar.header("Conversation History")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for i in range(len(st.session_state.chat_history) - 2, -1, -2):
            user_msg = st.session_state.chat_history[i]['content']
            bot_msg = st.session_state.chat_history[i + 1]['content']
            with st.sidebar.expander(f"User: {user_msg[:50]}..."):
                st.write(f"**Bot:** {bot_msg}")

# Main function to initialize the Streamlit app
def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = load_embeddings_offline("faiss_index")
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
            st.success("Embeddings successfully loaded offline!")
        except Exception as e:
            st.warning("No embeddings found. Please upload PDFs to generate embeddings.")
            st.session_state.conversation = None

    st.write(css, unsafe_allow_html=True)
    st.sidebar.subheader("Upload PDFs to Index")
    uploaded_pdfs = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs and st.sidebar.button("Process PDFs"):
        process_uploaded_pdfs(uploaded_pdfs)

    st.header("Chat with Pre-Computed Embeddings for Book Writing :books:")
    use_online_search = st.sidebar.checkbox("Enable Online Web Search")
    question = st.text_input("Ask a question:")
    if question:
        handle_question_with_search(question, use_online_search)
    display_memory_in_sidebar()

if __name__ == '__main__':
    main()