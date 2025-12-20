import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

def get_text_from_pdf(pdf_docs):
    raw_text = ""
    doc_stats = []
    for i, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        page_count = len(pdf_reader.pages)
        doc_text = ""
        for page in pdf_reader.pages:
            doc_text += page.extract_text()
        raw_text += doc_text
        doc_stats.append({
            "document": f"Document {i+1}",
            "pages": page_count,
            "characters": len(doc_text),
            "words": len(doc_text.split())
        })

    # Store document statistics in session state
    st.session_state.doc_stats = doc_stats
    return raw_text

def display_document_analytics():
    """Display analytics about processed documents"""
    if hasattr(st.session_state, 'doc_stats') and st.session_state.doc_stats:
        st.subheader(" Document Analytics")

        df = pd.DataFrame(st.session_state.doc_stats)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            st.metric("Total Pages", df['pages'].sum())
        with col3:
            st.metric("Total Words", f"{df['words'].sum():,}")
        with col4:
            st.metric("Total Characters", f"{df['characters'].sum():,}")

        # Document breakdown chart
        fig = px.bar(df, x='document', y='words', title='Word Count by Document')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_rag_pipeline():
    """Display RAG pipeline visualization"""
    st.subheader(" RAG Pipeline Architecture")

    # Create a visual pipeline
    pipeline_steps = [
        " PDF Upload",
        "✂ Text Chunking",
        "Embeddings",
        "Vector Store",
        "Retrieval",
        "LLM Generation"
    ]

    cols = st.columns(len(pipeline_steps))
    for i, (col, step) in enumerate(zip(cols, pipeline_steps)):
        with col:
            st.markdown(f"**{step}**")
            if i < len(pipeline_steps) - 1:
                st.markdown("⬇️")

    # Pipeline metrics
    if hasattr(st.session_state, 'processing_metrics'):
        st.write("**Processing Metrics:**")
        metrics = st.session_state.processing_metrics
        metric_cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(key.replace('_', ' ').title(), value)

def get_text_chunks(raw_text):
    start_time = time.time()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    processing_time = time.time() - start_time

    # Store chunking metrics
    chunk_stats = {
        "total_chunks": len(text_chunks),
        "avg_chunk_size": sum(len(chunk) for chunk in text_chunks) / len(text_chunks) if text_chunks else 0,
        "processing_time": f"{processing_time:.2f}s"
    }

    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {}
    st.session_state.processing_metrics.update(chunk_stats)

    return text_chunks

def get_vector_store(text_chunks):
    start_time = time.time()
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    processing_time = time.time() - start_time

    # Store vector metrics
    vector_stats = {
        "embedding_time": f"{processing_time:.2f}s",
        "vector_dimensions": 1536,  # OpenAI embedding dimension
        "indexed_chunks": len(text_chunks)
    }

    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {}
    st.session_state.processing_metrics.update(vector_stats)

    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1000)

    # Enhanced system template with better instructions
    system_template = """
    You are an intelligent document assistant with expertise in analyzing and discussing uploaded PDFs.
    Your role is to provide accurate, comprehensive, and helpful responses based on the document content.

    INSTRUCTIONS:
    - Use the retrieved context below to answer questions accurately
    - Provide detailed explanations when possible
    - If the answer isn't in the context, clearly state "This information is not available in the uploaded documents"
    - Cite specific sections or pages when relevant
    - Use markdown formatting for better readability
    - Be conversational yet professional

    RETRIEVED CONTEXT:
    {context}

    Remember: Your responses should be based solely on the provided context from the uploaded documents.
    """
    
    human_template = "{question}"
    
    # Create the chat prompt template
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5, "fetch_k": 20}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False
    )
    
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation is not None:
        # Record question timestamp
        question_time = datetime.now().strftime("%H:%M:%S")

        with st.chat_message("user"):
            st.write(f"**[{question_time}]** {user_question}")

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🔍 _Searching through documents..._")

            try:
                start_time = time.time()
                response = st.session_state.conversation({"question": user_question})
                response_time = time.time() - start_time

                answer = response["answer"]
                response_time_str = f"{response_time:.2f}s"

                # Add the current exchange to chat history with metadata
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": question_time
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "response_time": response_time_str,
                    "sources": len(response.get("source_documents", []))
                })

                # Display the response with metadata
                response_placeholder.markdown(f"{answer}\n\n_Response time: {response_time_str} | Sources: {len(response.get('source_documents', []))} documents_")

                # Enhanced source documents display
                if "source_documents" in response and response["source_documents"]:
                    with st.expander(f"📚 Source Documents ({len(response['source_documents'])} found)"):
                        for i, doc in enumerate(response["source_documents"]):
                            with st.container():
                                st.markdown(f"**📄 Source {i+1}**")

                                # Show relevance score if available
                                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                                    st.markdown(f"*Relevance Score: {doc.metadata['score']:.3f}*")

                                # Show content preview
                                content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                st.text_area(f"Content Preview {i+1}", content_preview, height=100, disabled=True)

                                if i < len(response["source_documents"]) - 1:
                                    st.divider()

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "error": True
                })

def main():
    load_dotenv()
    st.set_page_config(
        page_title="🎓 Advanced RAG System - PDF Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS for professional appearance
    st.markdown("""
    <style>
    /* Main title styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Chat container improvements */
    .chat-container {
        border-radius: 15px;
        margin-bottom: 15px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    
    /* Pipeline step styling */
    .pipeline-step {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 5px;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with professional styling
    st.markdown('<div class="main-header">🎓 Advanced RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Document Analysis with Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "show_analytics" not in st.session_state:
        st.session_state.show_analytics = False

    # Enhanced sidebar with better organization
    with st.sidebar:
        st.markdown("### 🔧 Control Panel")

        # Document upload section
        with st.container():
            st.markdown("#### 📁 Document Upload")
            pdf_docs = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF documents to analyze"
            )

            if pdf_docs:
                st.success(f"✅ {len(pdf_docs)} file(s) selected")

        # Control buttons
        st.markdown("#### ⚡ Actions")
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("🚀 Process", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", type="secondary", use_container_width=True)

        # Analytics toggle
        if st.session_state.processing_done:
            st.session_state.show_analytics = st.toggle("📊 Show Analytics", value=st.session_state.show_analytics)

        # Processing logic
        if clear_button:
            for key in ['messages', 'conversation', 'processing_done', 'doc_stats', 'processing_metrics']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if process_button:
            if pdf_docs:
                with st.spinner("🔄 Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Step 1: Extract text
                        status_text.text("📄 Extracting text from PDFs...")
                        progress_bar.progress(20)
                        raw_text = get_text_from_pdf(pdf_docs)
                        
                        # Step 2: Create chunks
                        status_text.text("✂️ Creating text chunks...")
                        progress_bar.progress(40)
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Step 3: Generate embeddings
                        status_text.text("🧮 Generating embeddings...")
                        progress_bar.progress(60)
                        vector_store = get_vector_store(text_chunks)
                        
                        # Step 4: Setup conversation
                        status_text.text("🤖 Setting up conversation chain...")
                        progress_bar.progress(80)
                        st.session_state.conversation = get_conversation_chain(vector_store)

                        # Complete
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        st.session_state.processing_done = True
                        
                        # Welcome message
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"""
 **Welcome to your RAG-powered PDF Assistant!**

I've successfully processed **{len(pdf_docs)} document(s)** with **{len(text_chunks)} text chunks**.

**What I can help you with:**
- Answer questions about your documents
- Summarize content
- Find specific information
- Compare information across documents

**How it works:**
1. 🔍 I search through your documents using semantic similarity
2. 📋 I retrieve the most relevant passages  
3. 🤖 I generate answers using the retrieved context

Feel free to ask me anything about your uploaded documents!
                            """,
                            "system_message": True
                        }]

                        time.sleep(1)  # Brief pause for better UX
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
            else:
                st.warning("⚠️ Please upload PDF files first")

        # System info
        with st.expander("ℹ️ System Information"):
            st.markdown("""
            **RAG Architecture:**
            - 🧠 Model: GPT-3.5-turbo
            - 📊 Embeddings: OpenAI text-embedding-ada-002
            - 🗄️ Vector Store: FAISS
            - 🔍 Retrieval: Top-5 similarity search
            - 💭 Memory: Conversation buffer
            
            **Features:**
            - Multi-document support
            - Real-time analytics
            - Source citation
            - Response timing
            - Conversation history
            """)
    
    # Main content area
    if st.session_state.show_analytics and st.session_state.processing_done:
        # Analytics Dashboard
        col1, col2 = st.columns([1, 1])

        with col1:
            display_document_analytics()

        with col2:
            display_rag_pipeline()

    # Chat Interface
    st.markdown("### 💬 Chat Interface")

    # Display chat history with enhanced formatting
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    timestamp = message.get("timestamp", "")
                    st.markdown(f"**[{timestamp}]** {message['content']}")

            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    if message.get("system_message"):
                        st.markdown(message["content"])
                    else:
                        response_time = message.get("response_time", "")
                        sources = message.get("sources", 0)

                        st.markdown(message["content"])

                        if response_time or sources:
                            st.caption(f"_⏱️ Response time: {response_time} | 📚 Sources: {sources} documents_")

    # Welcome message for new users
    if not st.session_state.processing_done and not st.session_state.messages:
        st.info("""
        👋 **Welcome to the Advanced RAG System!**
        
        📚 Upload your PDF documents using the sidebar and click "🚀 Process" to get started.
        
        This system uses state-of-the-art Retrieval-Augmented Generation to provide accurate answers based on your documents.
        """)

    # Chat input
    if user_question := st.chat_input(
        placeholder="Ask me anything about your documents...",
        disabled=not st.session_state.processing_done
    ):
        handle_user_input(user_question)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🎓 Advanced RAG System | Built with Streamlit, LangChain & OpenAI | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()