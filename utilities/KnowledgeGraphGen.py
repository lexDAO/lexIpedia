import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
from datetime import datetime
import PyPDF2
import xml.etree.ElementTree as ET


# File handling functions
def save_processed_data(nodes, relationships, original_text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert nodes and relationships to serializable format
    serializable_nodes = []
    serializable_relationships = []
    
    # Handle both tuple and Node object formats
    for node in nodes:
        if hasattr(node, 'id') and hasattr(node, 'type'):
            serializable_nodes.append({"id": str(node.id), "type": str(node.type)})
        else:
            serializable_nodes.append({"id": str(node[0]), "type": str(node[1])})
    
    # Handle both tuple and Relationship object formats
    for rel in relationships:
        if hasattr(rel, 'source') and hasattr(rel, 'type') and hasattr(rel, 'target'):
            serializable_relationships.append({
                "source": str(rel.source),
                "type": str(rel.type),
                "target": str(rel.target)
            })
        else:
            serializable_relationships.append({
                "source": str(rel[0]),
                "type": str(rel[1]),
                "target": str(rel[2])
            })
    
    data = {
        'nodes': serializable_nodes,
        'relationships': serializable_relationships,
        'original_text': original_text
    }
    
    # Save both JSON and pretty-printed JSON
    filename = f'graph_data_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create a more human-readable version
    readable_filename = f'readable_graph_{timestamp}.txt'
    with open(readable_filename, 'w') as f:
        f.write("KNOWLEDGE GRAPH ENTITIES:\n")
        f.write("=======================\n\n")
        for node in serializable_nodes:
            f.write(f"Entity: {node['id']} (Type: {node['type']})\n")
        
        f.write("\nKNOWLEDGE GRAPH RELATIONSHIPS:\n")
        f.write("=============================\n\n")
        for rel in serializable_relationships:
            f.write(f"{rel['source']} → {rel['type']} → {rel['target']}\n")
    
    return filename, readable_filename
def extract_text_from_xml(xml_file):
    """Extract text from an XML file"""
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract text from all elements
        def extract_text_from_element(element):
            text = ""
            # Add element's text
            if element.text and element.text.strip():
                text += element.text.strip() + "\n"
            
            # Recursively extract text from child elements
            for child in element:
                text += extract_text_from_element(child)
            
            # Add tail text (text after the element)
            if element.tail and element.tail.strip():
                text += element.tail.strip() + "\n"
            
            return text
        
        return extract_text_from_element(root)
    except Exception as e:
        st.error(f"Error parsing XML file: {str(e)}")
        return ""

def load_processed_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert to the format expected by the rest of the code
    nodes = [(node['id'], node['type']) for node in data['nodes']]
    relationships = [(rel['source'], rel['type'], rel['target']) for rel in data['relationships']]
    
    return nodes, relationships, data.get('original_text', '')

def list_saved_files():
    return [f for f in os.listdir('.') if f.startswith('graph_data_') and f.endswith('.json')]

class Neo4jImporter:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
    def close(self):
        self.driver.close()

    def create_constraints(self, nodes):
        with self.driver.session(database=self.database) as session:
            for node in set(node[1] for node in nodes):
                node_type = node.replace(" ", "_")
                session.run(f"""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{node_type}`) 
                    REQUIRE n.id IS UNIQUE
                """)

    def clear_existing_data(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_nodes_and_relationships(self, nodes, relationships):
        with self.driver.session(database=self.database) as session:
            for node_id, node_type in nodes:
                node_type = node_type.replace(" ", "_")
                session.run(f"""
                    MERGE (n:`{node_type}` {{id: $id}})
                """, id=node_id)

            for source, rel_type, target in relationships:
                rel_type = rel_type.replace(" ", "_").upper()
                session.run(f"""
                    MATCH (a {{id: $source}})
                    MATCH (b {{id: $target}})
                    MERGE (a)-[:`{rel_type}`]->(b)
                """, source=source, target=target)

    def get_all_data(self):
        with self.driver.session(database=self.database) as session:
            nodes_result = session.run("""
                MATCH (n)
                RETURN n.id as id, labels(n) as labels
            """)
            nodes = [(record["id"], record["labels"][0]) for record in nodes_result]
            
            rels_result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN n.id as source, type(r) as type, m.id as target
            """)
            relationships = [(record["source"], record["type"], record["target"]) 
                           for record in rels_result]
            
            return nodes, relationships

    def export_relationships_as_csv(self, filename="relationships_export.csv"):
        """Export all relationships in the graph to a CSV file"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN 
                    labels(n)[0] as source_type,
                    n.id as source_id, 
                    type(r) as relationship, 
                    labels(m)[0] as target_type,
                    m.id as target_id
            """)
            
            # Convert to DataFrame for easy CSV export
            records = []
            for record in result:
                records.append({
                    'subject_type': record['source_type'],
                    'subject': record['source_id'],
                    'predicate': record['relationship'],
                    'object_type': record['target_type'],
                    'object': record['target_id']
                })
            
            df = pd.DataFrame(records)
            df.to_csv(filename, index=False)
            return filename

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_csv(csv_file):
    """Extract text from a CSV file"""
    df = pd.read_csv(csv_file)
    # Concatenate all text columns into a single string
    text = ""
    for column in df.columns:
        text += f"{column}:\n"
        text += "\n".join(df[column].astype(str).tolist())
        text += "\n\n"
    return text

def chunk_text(text, chunk_size=2000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def process_langchain_output(graph_documents):
    nodes_dict = {}
    relationships_dict = {}
    
    try:
        for doc in graph_documents:
            for node in doc.nodes:
                # Convert Node object attributes to strings
                node_id = str(node.id) if hasattr(node, 'id') else str(node)
                node_type = str(node.type) if hasattr(node, 'type') else "Unknown"
                nodes_dict[node_id] = (node_id, node_type)
            
            for rel in doc.relationships:
                # Convert Relationship object attributes to strings
                source = str(rel.source) if hasattr(rel, 'source') else str(rel[0])
                rel_type = str(rel.type) if hasattr(rel, 'type') else str(rel[1])
                target = str(rel.target) if hasattr(rel, 'target') else str(rel[2])
                
                rel_key = f"{source}|{rel_type}|{target}"
                relationships_dict[rel_key] = (source, rel_type, target)
        
        return list(nodes_dict.values()), list(relationships_dict.values())
    except Exception as e:
        st.error(f"Error processing LangChain output: {str(e)}")
        st.error(f"Debug info - doc structure: {str(graph_documents[0]) if graph_documents else 'No documents'}")
        return [], []

def visualize_simplified_graph(nodes, relationships, max_nodes=20):
    """Create a simplified graph visualization with node limits"""
    G = nx.DiGraph()
    
    # Limit the number of nodes for better visualization
    display_nodes = nodes[:max_nodes]
    
    # Add nodes to graph
    for node_id, node_type in display_nodes:
        G.add_node(node_id, node_type=node_type)
    
    # Only add relationships between displayed nodes
    display_node_ids = [node[0] for node in display_nodes]
    for source, rel_type, target in relationships:
        if source in display_node_ids and target in display_node_ids:
            G.add_edge(source, target, relationship=rel_type)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes with different colors based on type
    node_types = set(node[1] for node in display_nodes)
    colors = plt.cm.tab10(range(len(node_types)))
    type_to_color = dict(zip(node_types, colors))
    
    # Draw nodes
    for node_type in node_types:
        nodes_of_type = [node[0] for node in display_nodes if node[1] == node_type]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=nodes_of_type,
                              node_color=[type_to_color[node_type]],
                              node_size=1000, 
                              alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                label=f"{node_type}", 
                                markerfacecolor=type_to_color[node_type], 
                                markersize=10) 
                    for node_type in node_types]
    plt.legend(handles=legend_handles, loc='upper right')
    
    plt.title("Simplified Knowledge Graph Visualization")
    plt.axis('off')
    return plt

def main():
    st.title("Knowledge Graph Generator with JSON Output")
   
    # Database connection parameters
    with st.sidebar.expander("Neo4j Connection (Optional)"):
        use_neo4j = st.checkbox("Use Neo4j Database", value=False)
        uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
        user = st.text_input("Username", "neo4j")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database", "neo4j")

    # OpenAI settings
    st.sidebar.header("OpenAI Settings")
    st.sidebar.write("""
    This application helps you convert legal code in many different formats into knowledge graphs. 
    The relationships and nodes are downloadable as CSV's or JSON files. 
    Currently, the visualization does not function as it should, but this is not an issue with the knowledge graph output.  
    """)
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    

    # Model selection
    model_options = {
        "GPT-4 (More accurate)": "gpt-4",
        "GPT-3.5 Turbo (Faster)": "gpt-3.5-turbo"
    }
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    # Additional settings
    with st.sidebar.expander("Advanced Settings"):
        chunk_size = st.slider(
            "Chunk Size (characters)", 
            min_value=1000, 
            max_value=10000, 
            value=3000,
            step=500
        )
        
        max_display_nodes = st.slider(
            "Max Nodes to Display", 
            min_value=5, 
            max_value=50, 
            value=20
        )

    # Setup Neo4j if selected
    importer = None
    if use_neo4j:
        try:
            importer = Neo4jImporter(uri, user, password, database)
        except Exception as e:
            st.sidebar.error(f"Neo4j connection error: {str(e)}")

    # Load previous results section
    st.sidebar.header("Load Previous Results")
    saved_files = list_saved_files()
    if saved_files:
        selected_file = st.sidebar.selectbox(
            "Select previously processed data",
            saved_files,
            format_func=lambda x: f"{x[10:-5]}"
        )
        
        if st.sidebar.button("Load Selected Data"):
            nodes, relationships, original_text = load_processed_data(selected_file)
            st.success(f"Loaded data from {selected_file}")
            
            # Create visualization
            with st.expander("Graph Visualization", expanded=True):
                st.warning("This is a simplified visualization. For a complete view, check the JSON output.")
                plt = visualize_simplified_graph(nodes, relationships, max_display_nodes)
                st.pyplot(plt)
            
            # Show data in different formats
            tab1, tab2, tab3 = st.tabs(["JSON", "Entities", "Relationships"])
            
            with tab1:
                # Convert back to JSON format
                json_data = {
                    'nodes': [{'id': n[0], 'type': n[1]} for n in nodes],
                    'relationships': [{'source': r[0], 'type': r[1], 'target': r[2]} for r in relationships]
                }
                st.json(json_data)
                
                # Export as JSON
                json_string = json.dumps(json_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_string,
                    file_name="knowledge_graph.json",
                    mime="application/json"
                )
            
            with tab2:
                nodes_df = pd.DataFrame(nodes, columns=['ID', 'Type'])
                st.dataframe(nodes_df)
            
            with tab3:
                rels_df = pd.DataFrame(relationships, columns=['Subject', 'Predicate', 'Object'])
                st.dataframe(rels_df)

    # Input section
    st.subheader("Input Text for Graph Generation")
    text_input_method = st.radio(
        "Choose input method",
        ["Enter text directly", "Upload a file"]
    )
    
    file_content = None
    if text_input_method == "Enter text directly":
        text_input = st.text_area("Enter text", height=150)
    else:  # Upload a file
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['txt', 'csv', 'pdf', 'xml']
        )
        
        if uploaded_file is not None:
            # Process the file based on its type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'txt':
                file_content = uploaded_file.read().decode()
                st.text_area("File Content Preview", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150)
            
            elif file_type == 'csv':
                file_content = extract_text_from_csv(uploaded_file)
                # Preview the data in a more structured way
                df = pd.read_csv(uploaded_file)
                st.write("CSV Data Preview:")
                st.dataframe(df.head())
                
            elif file_type == 'pdf':
                # Reset stream position
                uploaded_file.seek(0)
                file_content = extract_text_from_pdf(uploaded_file)
                st.text_area("PDF Content Preview", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150)

            elif file_type == 'xml':
                # Reset stream position
                uploaded_file.seek(0)
                file_content = extract_text_from_xml(uploaded_file)
                st.text_area("XML Content Preview", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150)

    if st.button("Generate Knowledge Graph"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            content = None
            if text_input_method == "Enter text directly":
                content = text_input
            else:
                content = file_content
            
            if not content:
                st.error("Please either enter text or upload a file.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Split text into chunks
                chunks = chunk_text(content, chunk_size)
                total_chunks = len(chunks)
                
                status_text.text(f"Processing text in {total_chunks} chunks...")
                
                # Initialize LangChain components
                llm = ChatOpenAI(
                    temperature=0, 
                    model=model_options[selected_model],
                    api_key=openai_api_key
                )
                llm_transformer = LLMGraphTransformer(llm=llm)
                
                all_graph_documents = []
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Processing chunk {i+1} of {total_chunks}...")
                    progress_bar.progress((i + 1) / total_chunks)
                    
                    # Process chunk
                    documents = [Document(page_content=chunk)]
                    graph_documents = llm_transformer.convert_to_graph_documents(documents)
                    all_graph_documents.extend(graph_documents)
                
                # Process all results
                nodes, relationships = process_langchain_output(all_graph_documents)
                
                if nodes and relationships:
                    # Save the processed data
                    json_file, readable_file = save_processed_data(nodes, relationships, content)
                    st.success(f"Saved processed data to {json_file} and {readable_file}")
                    
                    # Import to Neo4j if selected
                    if use_neo4j and importer:
                        status_text.text("Importing to Neo4j...")
                        importer.create_constraints(nodes)
                        importer.clear_existing_data()
                        importer.create_nodes_and_relationships(nodes, relationships)
                    
                    # Create visualization
                    status_text.text("Generating visualization...")
                    with st.expander("Graph Visualization", expanded=True):
                        st.warning("This is a simplified visualization showing only the first few nodes. For a complete view, check the JSON output.")
                        plt = visualize_simplified_graph(nodes, relationships, max_display_nodes)
                        st.pyplot(plt)
                    
                    # Show data in different formats
                    tab1, tab2, tab3, tab4 = st.tabs(["JSON", "Readable Format", "Entities", "Relationships"])
                    
                    with tab1:
                        # Convert to JSON format
                        json_data = {
                            'nodes': [{'id': n[0], 'type': n[1]} for n in nodes],
                            'relationships': [{'source': r[0], 'type': r[1], 'target': r[2]} for r in relationships]
                        }
                        st.json(json_data)
                        
                        # Export as JSON
                        json_string = json.dumps(json_data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_string,
                            file_name="knowledge_graph.json",
                            mime="application/json"
                        )
                    
                    with tab2:
                        # Display in a more readable format
                        st.subheader("Entities")
                        for node_id, node_type in nodes:
                            st.write(f"• {node_id} (Type: {node_type})")
                        
                        st.subheader("Relationships")
                        for source, rel_type, target in relationships:
                            st.write(f"• {source} → {rel_type} → {target}")
                        
                        # Generate a downloadable readable format
                        readable_content = "KNOWLEDGE GRAPH ENTITIES:\n"
                        readable_content += "=======================\n\n"
                        for node_id, node_type in nodes:
                            readable_content += f"Entity: {node_id} (Type: {node_type})\n"
                        
                        readable_content += "\nKNOWLEDGE GRAPH RELATIONSHIPS:\n"
                        readable_content += "=============================\n\n"
                        for source, rel_type, target in relationships:
                            readable_content += f"{source} → {rel_type} → {target}\n"
                        
                        st.download_button(
                            label="Download Readable Format",
                            data=readable_content,
                            file_name="knowledge_graph_readable.txt",
                            mime="text/plain"
                        )
                    
                    with tab3:
                        nodes_df = pd.DataFrame(nodes, columns=['ID', 'Type'])
                        st.dataframe(nodes_df)
                    
                    with tab4:
                        rels_df = pd.DataFrame(relationships, columns=['Subject', 'Predicate', 'Object'])
                        st.dataframe(rels_df)
                    
                    status_text.text("Processing complete!")
                    progress_bar.progress(100)

if __name__ == "__main__":
    main()
