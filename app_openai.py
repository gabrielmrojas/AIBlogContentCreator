__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import tempfile
import fitz
import streamlit as st
import os
import time
import chromadb
from fpdf import FPDF
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import uuid
import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import slide

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="AI Blog Creator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


class AIBlogCreator:
    # Update the ChromaDB client initialization in your AIBlogCreator class
    def __init__(self, api_key=None, collection_name="blog_content"):
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model_name = "gpt-4o-mini"  # Default to gpt-4o-mini

        # Set up embeddings with explicit model name
        self.embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            model="text-embedding-ada-002"  # Explicitly set the embedding model
        )

        # Set up ChromaDB with the new client approach
        try:
            # Create a temporary directory if it doesn't exist
            os.makedirs("./chroma_db", exist_ok=True)

            # Generate a unique collection name with OpenAI embedding identifier
            self.collection_name = f"{collection_name}_openai_1536"

            # Use the new client initialization
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
            )

            # Get or create the native ChromaDB collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # Set up vector store with explicit embedding function
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client=self.chroma_client
            )
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            # Fallback to in-memory client with a unique name
            try:
                self.chroma_client = chromadb.EphemeralClient()
                self.collection_name = f"temp_collection_openai_1536_{uuid.uuid4().hex[:8]}"

                # Create the native ChromaDB collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                # Set up vector store with fallback collection
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    client=self.chroma_client
                )
            except Exception as e2:
                st.error(f"Error with fallback ChromaDB: {str(e2)}")
                raise e2

    def generate_blog(self, topic, word_count=500, use_retrieval=True):
        """Generate a blog post on the given topic"""
        client = openai.OpenAI(api_key=self.api_key)

        # Prepare the prompt
        system_prompt = (
            "You are an expert blog writer capable of creating engaging, informative, and well-structured blog posts. "
            "Your task is to write a blog post on the provided topic. "
            "The blog should be organized with a clear introduction, body, and conclusion. "
            "Incorporate relevant information and maintain a conversational yet professional tone."
        )

        user_prompt = f"Write a blog post about '{topic}' with approximately {word_count} words."

        # Add context from the database if retrieval is enabled
        context = ""
        if use_retrieval and self.collection.count() > 0:
            # Query the collection for relevant content
            try:
                query_embedding = self.embeddings.embed_query(topic)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3
                )

                if results and 'documents' in results and results['documents']:
                    # Extract documents and create context
                    documents = results['documents'][0]
                    if documents:
                        context = "Here is some additional context that might be helpful:\n\n"
                        for doc in documents:
                            context += f"{doc}\n\n"

                if context:
                    user_prompt += f"\n\n{context}"
            except Exception as e:
                st.warning(f"Error retrieving context: {str(e)}")

        # Generate the blog content
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )

            blog_content = response.choices[0].message.content

            # Store the generated content in the vector database using the vector_store instead of direct collection access
            if len(blog_content) > 0:
                try:
                    # Use vector_store's add_texts method to ensure embedding consistency
                    self.vector_store.add_texts(
                        texts=[blog_content],
                        metadatas=[{"topic": topic, "timestamp": str(time.time())}]
                    )
                except Exception as e:
                    st.warning(f"Error storing blog in database: {str(e)}")

            return blog_content
        except Exception as e:
            st.error(f"Error generating blog: {str(e)}")
            return f"An error occurred while generating the blog: {str(e)}"

    def generate_presentation_content(self, blog_content):
        """Generate presentation slide content from blog content"""
        client = openai.OpenAI(api_key=self.api_key)

        system_prompt = (
            "You are an expert at summarizing content for presentations. "
            "Your task is to review the blog content and create a slide presentation summary. "
            "Break the content into 5-7 slide sections. Each slide should have:"
            "- A slide title (in bold with format: **Slide N: Title**)"
            "- 3 bullet points summarizing the key information (with format: * Bullet point text)"
            "The bullet points should be concise but informative."
        )

        user_prompt = f"Create a presentation summary for the following blog content:\n\n{blog_content}"

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )

            presentation_content = response.choices[0].message.content
            return presentation_content
        except Exception as e:
            st.error(f"Error generating presentation content: {str(e)}")
            return f"An error occurred while generating the presentation: {str(e)}"


def create_pdf_from_markdown(content, title):
    """Generate a PDF document from markdown content"""
    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()

    # Use Helvetica instead of Arial
    pdf.set_font("Helvetica", size=12)

    # Add title
    pdf.set_font("Helvetica", style="B", size=16)
    # Set a specific width less than the page width
    page_width = pdf.w - 20  # Leave some margin
    pdf.cell(page_width, 10, text=title, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # Reset font
    pdf.set_font("Helvetica", size=12)

    # Process content by lines
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue

        # Handle headers
        if line.startswith('## '):
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(page_width, 10, text=line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=12)
        elif line.startswith('### '):
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(page_width, 10, text=line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=12)
        # Handle normal text
        else:
            # Ensure text is Latin-1 encoded or replace problematic chars
            try:
                safe_line = line.encode('latin-1', errors='replace').decode('latin-1')
                # Use width parameter for multi_cell to ensure text fits
                pdf.multi_cell(page_width, 5, text=safe_line)
            except Exception as e:
                # Skip problematic lines instead of failing
                pdf.multi_cell(page_width, 5, text=f"[Content omitted due to encoding issue]")

    # Return the PDF as bytes
    try:
        return pdf.output()
    except Exception as e:
        # Fallback to simpler content if needed
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text="Error creating formatted PDF. Please try a different format.")
        return pdf.output()


def create_simple_pdf(title, content):
    """Create a simple PDF with title and content"""
    pdf = FPDF()
    pdf.add_page()

    # Set font
    pdf.set_font("Helvetica", size=12)

    # Calculate safe width
    page_width = pdf.w - 20  # Leave margins

    # Ensure title and content are Latin-1 compatible
    try:
        safe_title = title.encode('latin-1', errors='replace').decode('latin-1')
        safe_content = content.encode('latin-1', errors='replace').decode('latin-1')

        # Add title and content
        pdf.cell(page_width, 10, text=safe_title, new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)

        # Add content with proper width
        pdf.multi_cell(page_width, 5, text=safe_content)
    except Exception as e:
        # Fallback to very simple content
        pdf.cell(page_width, 10, text="PDF Content Error", new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(page_width, 5, text=f"Error creating PDF: {str(e)}")

    # Return PDF as bytes
    return pdf.output()

def main():
    st.title("AI Blog Content Creator")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Initialize session state
    if 'blog_creator' not in st.session_state:
        try:
            st.session_state.blog_creator = AIBlogCreator(api_key=api_key)
        except ValueError as e:
            st.error(str(e))
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if api_key:
                try:
                    st.session_state.blog_creator = AIBlogCreator(api_key=api_key)
                    st.success("API key accepted! You can now generate blogs.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error initializing with provided API key: {str(e)}")
                    return
            return

    # Create tabs
    tab1, tab2 = st.tabs(["Generate Content Blog", "Generated Content"])

    # Tab 1: Blog Generation
    with tab1:
        st.header("Generate Content Blog")

        # Input section for blog generation
        col1, col2 = st.columns([2, 1])

        with col1:
            topic = st.text_input("Blog Topic:", placeholder="Enter a topic for your blog post")

        with col2:
            word_count = st.number_input("Word Count:",
                                         min_value=100,
                                         max_value=2000,
                                         value=500,
                                         step=100)

        use_retrieval = st.checkbox("Use document knowledge base", value=True,
                                    help="When checked, the AI will use information from uploaded PDFs to enhance the blog")

        if st.button("Generate Blog"):
            if not topic:
                st.warning("Please enter a topic for your blog post.")
            else:
                with st.spinner("Generating blog..."):
                    try:
                        blog_content = st.session_state.blog_creator.generate_blog(
                            topic=topic,
                            word_count=word_count,
                            use_retrieval=use_retrieval
                        )

                        # Store in session state for the second tab
                        st.session_state.blog_topic = topic
                        st.session_state.blog_content = blog_content

                        st.success(
                            "Blog generated successfully! Go to the 'Generated Content' tab to view and download.")
                    except Exception as e:
                        st.error(f"Error generating blog: {str(e)}")

        # Document upload and processing section
        st.header("Upload PDF Documents")
        st.markdown("Upload PDF documents to enhance the AI's knowledge base for blog generation.")

        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        uploaded_file = st.file_uploader("Limit 5MB per file • PDF", type="pdf")

        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the 5MB limit. Current size: {file_size / (1024 * 1024):.2f}MB")
            else:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        # Create a temporary file to store the uploaded content
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                            tmp_pdf.write(uploaded_file.getbuffer())
                            pdf_path = tmp_pdf.name

                        try:
                            # Extract text from PDF
                            doc = fitz.open(pdf_path)
                            text = ""
                            for page in doc:
                                text += page.get_text()

                            # Convert to markdown (simplified)
                            md_text = text.replace("\n\n", "\n\n## ")
                            md_text = f"# Document: {uploaded_file.name}\n\n{md_text}"

                            # Split text into chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len
                            )
                            chunks = text_splitter.split_text(md_text)

                            # Store in vector db with metadata
                            metadata = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

                            # Add to vector store
                            st.session_state.blog_creator.vector_store.add_texts(
                                texts=chunks,
                                metadatas=metadata
                            )

                            st.success(f"Document '{uploaded_file.name}' processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(pdf_path):
                                os.unlink(pdf_path)

    # Tab 2: Generated Content
    with tab2:
        st.header("Generated Blog Content")

        # Check if a blog has been generated
        if 'blog_content' in st.session_state:
            blog_topic = st.session_state.blog_topic
            blog_content = st.session_state.blog_content

            st.subheader(blog_topic)
            st.markdown(blog_content)

            # Download options
            st.subheader("Download Options")

            col1, col2 = st.columns(2)

            with col1:
                # Markdown download
                st.download_button(
                    label="Download as Markdown",
                    data=blog_content,
                    file_name=f"{blog_topic.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

            with col2:
                # PDF download
                try:
                    pdf_bytes = create_pdf_from_markdown(blog_content, blog_topic)

                    # Convert bytearray to bytes if needed
                    if isinstance(pdf_bytes, bytearray):
                        pdf_bytes = bytes(pdf_bytes)

                    st.download_button(
                        label="Download as PDF",
                        data=pdf_bytes,
                        file_name=f"{blog_topic.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error creating PDF: {str(e)}")

            # Create a presentation PDF
            st.markdown("### Presentation Slides")
            st.write("Generate a PDF presentation that summarizes the blog content:")

            try:
                # Create an expander to show the progress and content
                with st.expander("Presentation Generation Progress", expanded=True):
                    progress_placeholder = st.empty()
                    progress_placeholder.info("Step 1/3: Analyzing blog content...")

                    # Generate the slide content
                    slide_content = st.session_state.blog_creator.generate_presentation_content(blog_content)

                    # Update progress
                    progress_placeholder.info("Step 2/3: Formatting slides...")

                    # Preview section
                    st.subheader("Slide Content Preview")
                    st.markdown(slide_content)
                    st.markdown("---")
                    st.markdown("*Formatting this content into presentation slides...*")

                    # Create the PDF
                    try:
                        presentation_bytes = slide.create_presentation_pdf(blog_topic, slide_content)

                        # Convert bytearray to bytes if needed
                        if isinstance(presentation_bytes, bytearray):
                            presentation_bytes = bytes(presentation_bytes)

                        # Final update
                        progress_placeholder.success("Step 3/3: PDF creation complete!")

                        # Show the download button outside the expander
                        st.success("Presentation created successfully!")
                        st.download_button(
                            label="Download Presentation PDF",
                            data=presentation_bytes,
                            file_name=f"{blog_topic.replace(' ', '_')}_presentation.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        progress_placeholder.error(f"Error in PDF creation: {str(e)}")
                        st.error(f"Could not create presentation PDF: {str(e)}")

                # Add a preview note
                with st.expander("About this presentation", expanded=False):
                    st.info("The presentation includes:\n"
                            "- A title slide\n"
                            "- 5-7 content slides with key points from each section\n"
                            "- A closing slide")

                    # Show slide example
                    st.markdown("#### Example Slide Format")
                    st.markdown("""
                    Each slide contains:
                    - A clear section title
                    - 2-3 bullet points summarizing key information
                    - Clean, professional formatting
                    """)

            except Exception as e:
                st.error(f"Error creating presentation: {str(e)}")
                st.info("Please install ReportLab with 'pip install reportlab' to enable presentation generation")

        else:
            st.info("No blog has been generated yet. Go to the 'Generate Blog' tab to create one.")



if __name__ == "__main__":
    main()