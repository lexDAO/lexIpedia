import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os
import pandas as pd
import base64
import re
from io import BytesIO
import streamlit.components.v1 as components

class KnowledgeGraph:
    def __init__(self, json_data):
        self.data = json_data
        self.graph = nx.DiGraph()
        self.parse_knowledge_graph()
    
    def parse_knowledge_graph(self):
        """Parse JSON knowledge graph into a NetworkX graph"""
        # Get nodes and relationships from the data
        nodes = self.data.get("nodes", [])
        relationships = self.data.get("relationships", [])
        
        # Add all nodes first
        for node in nodes:
            try:
                # Parse node information - handle string format or dict format
                if isinstance(node, dict):
                    node_id = node.get("id", "")
                    node_type = node.get("type", "")
                    node_properties = node.get("properties", {})
                elif isinstance(node, str):
                    # Try to parse string format: "id='X' type='Y' properties={}"
                    node_id = self._extract_value(node, "id")
                    node_type = self._extract_value(node, "type")
                    node_properties = {}
                else:
                    continue
                
                # Skip if no valid ID
                if not node_id:
                    continue
                
                # Add node to graph
                self.graph.add_node(
                    node_id, 
                    type=node_type,
                    label=node_id,
                    **node_properties
                )
            except Exception as e:
                st.warning(f"Error parsing node: {node}. Error: {str(e)}")
        
        # Add relationships
        for rel in relationships:
            try:
                if isinstance(rel, list) and len(rel) >= 3:
                    # Format: [source_node, relationship_type, target_node]
                    source_info = rel[0]
                    relationship_type = rel[1]
                    target_info = rel[2]
                    
                    # Extract source and target IDs
                    if isinstance(source_info, dict):
                        source_id = source_info.get("id", "")
                    elif isinstance(source_info, str):
                        source_id = self._extract_value(source_info, "id")
                    else:
                        continue
                    
                    if isinstance(target_info, dict):
                        target_id = target_info.get("id", "")
                    elif isinstance(target_info, str):
                        target_id = self._extract_value(target_info, "id")
                    else:
                        continue
                    
                    # Add edge if both source and target exist
                    if source_id and target_id:
                        # Add any missing nodes
                        if source_id not in self.graph:
                            source_type = ""
                            if isinstance(source_info, dict):
                                source_type = source_info.get("type", "")
                            elif isinstance(source_info, str):
                                source_type = self._extract_value(source_info, "type")
                            self.graph.add_node(source_id, type=source_type, label=source_id)
                        
                        if target_id not in self.graph:
                            target_type = ""
                            if isinstance(target_info, dict):
                                target_type = target_info.get("type", "")
                            elif isinstance(target_info, str):
                                target_type = self._extract_value(target_info, "type")
                            self.graph.add_node(target_id, type=target_type, label=target_id)
                        
                        # Add the edge with relationship type
                        self.graph.add_edge(source_id, target_id, type=relationship_type)
                elif isinstance(rel, str):
                    # Try to parse from a single string
                    parts = rel.split('"')
                    if len(parts) >= 5:
                        source_info = parts[0]
                        relationship_type = parts[1]
                        target_info = parts[2]
                        
                        source_id = self._extract_value(source_info, "id")
                        target_id = self._extract_value(target_info, "id")
                        
                        if source_id and target_id:
                            # Add nodes if needed
                            if source_id not in self.graph:
                                source_type = self._extract_value(source_info, "type")
                                self.graph.add_node(source_id, type=source_type, label=source_id)
                            
                            if target_id not in self.graph:
                                target_type = self._extract_value(target_info, "type")
                                self.graph.add_node(target_id, type=target_type, label=target_id)
                            
                            # Add edge
                            self.graph.add_edge(source_id, target_id, type=relationship_type)
            except Exception as e:
                st.warning(f"Error parsing relationship: {rel}. Error: {str(e)}")
    
    def _extract_value(self, text, key):
        """Extract a value from a string like id='X' type='Y'"""
        pattern = f"{key}='([^']*)'|{key}=\"([^\"]*)\""
        match = re.search(pattern, text)
        if match:
            return match.group(1) or match.group(2)
        return ""
    
    def to_powl(self):
        """Convert knowledge graph to POWL model"""
        powl_model = POWLModel(self.graph)
        return powl_model

class POWLModel:
    def __init__(self, graph):
        self.graph = graph
        self.tasks = []
        self.control_flow = []
        self.data_flow = []
        
        # Map relationship types to flow types
        self.control_flow_types = [
            "FOLLOWS", "PRECEDES", "NEXT", "SEQUENCE", 
            "REQUIRES", "DEPENDS_ON", "TRIGGERS",
            "APPROVES", "AUTHORIZES", "DECIDES"
        ]
        
        self.data_flow_types = [
            "USES", "PRODUCES", "CONSUMES", "CREATES", "DATA",
            "INPUTS", "OUTPUTS", "READS", "WRITES", "MODIFIES",
            "DEFINED_AS", "REFERENCES"
        ]
        
        self.process_graph()
    
    def process_graph(self):
        """Process the graph to identify tasks, control flow, and data flow"""
        # Identify tasks (activities)
        # Consider organization types as potential workflow participants
        task_types = ["ACTIVITY", "TASK", "PROCESS", "STEP", "ACTION", 
                     "ORGANIZATION", "DEPARTMENT", "ENTITY", "BOARD", 
                     "COMMITTEE", "AUTHORITY"]
        
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "").upper()
            if any(task_type in node_type for task_type in task_types):
                self.tasks.append(node)
        
        # If no explicit task types are found, look for nodes that initiate relationships
        if not self.tasks:
            # Find nodes that have outgoing edges (potential tasks)
            for node in self.graph.nodes():
                if self.graph.out_degree(node) > 0:
                    self.tasks.append(node)
        
        # Identify control flow and data flow relationships
        for source, target, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("type", "").upper()
            
            if edge_type in self.control_flow_types:
                self.control_flow.append((source, target))
            elif edge_type in self.data_flow_types:
                self.data_flow.append((source, target))
            else:
                # If relationship type is not recognized, default to control flow
                # if both source and target are tasks
                if source in self.tasks and target in self.tasks:
                    self.control_flow.append((source, target))
                else:
                    # Otherwise, consider it data flow
                    self.data_flow.append((source, target))
    
    def to_petri_net(self):
        """Convert POWL model to Petri Net representation"""
        petri_net = nx.DiGraph()
        
        # Create places for each task
        for task in self.tasks:
            # Each task gets an input and output place
            in_place = f"p_in_{task}"
            out_place = f"p_out_{task}"
            petri_net.add_node(in_place, type="place", label=f"In: {task}")
            petri_net.add_node(out_place, type="place", label=f"Out: {task}")
            petri_net.add_node(task, type="transition", label=task)
            
            # Connect places to transition
            petri_net.add_edge(in_place, task)
            petri_net.add_edge(task, out_place)
        
        # Connect control flow between tasks
        for source, target in self.control_flow:
            if source in self.tasks and target in self.tasks:
                source_out = f"p_out_{source}"
                target_in = f"p_in_{target}"
                petri_net.add_edge(source_out, target_in)
        
        return petri_net
    
    def to_dict(self):
        """Convert POWL model to dictionary representation"""
        return {
            "tasks": self.tasks,
            "control_flow": self.control_flow,
            "data_flow": self.data_flow
        }
    
    def to_json(self):
        """Convert POWL model to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

def visualize_graph(graph, height=800, physics_enabled=True):
    """Create interactive visualization of the graph"""
    # Create a pyvis network
    net = Network(height=f"{height}px", width="100%", notebook=True, directed=True)
    
    # Add nodes
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "default")
        label = attrs.get("label", str(node))
        
        # Truncate very long labels
        display_label = label if len(str(label)) < 30 else f"{str(label)[:27]}..."
        
        # Set node color based on type
        if node_type == "place":
            color = "#6929c4"  # Purple for places
            shape = "dot"
        elif node_type == "transition":
            color = "#1192e8"  # Blue for transitions
            shape = "box"
        elif node_type.upper() in ["ACTIVITY", "TASK", "PROCESS", "STEP", "ACTION"]:
            color = "#fa4d56"  # Red for activities/tasks
            shape = "box"
        elif node_type.upper() in ["ORGANIZATION", "DEPARTMENT", "ENTITY", "BOARD", "COMMITTEE"]:
            color = "#08bdba"  # Teal for organizations
            shape = "diamond"
        elif node_type.upper() in ["DOCUMENT", "FORM", "REPORT", "CODE"]:
            color = "#33b1ff"  # Light blue for documents
            shape = "file"
        elif node_type.upper() in ["CONCEPT", "TERM", "DEFINITION"]:
            color = "#d2a106"  # Gold for concepts
            shape = "ellipse"
        else:
            color = "#8a3ffc"  # Default color
            shape = "ellipse"
        
        # Add the node with appropriate attributes
        net.add_node(node, label=display_label, title=f"{node_type}: {label}", 
                    color=color, shape=shape)
    
    # Add edges
    for source, target, attrs in graph.edges(data=True):
        edge_type = attrs.get("type", "")
        title = edge_type if edge_type else "connects"
        
        # Choose edge color based on relationship type
        if edge_type.upper() in ["FOLLOWS", "PRECEDES", "NEXT", "SEQUENCE", "APPROVES"]:
            color = "#fa4d56"  # Red for control flow
        else:
            color = "#8a3ffc"  # Purple for data flow
        
        net.add_edge(source, target, title=title, label=edge_type, color=color)
    
    # Set physics layout options
    net.toggle_physics(physics_enabled)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "font": {
          "size": 8,
          "align": "middle"
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      }
    }
    """)
    
    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        path = tmpfile.name
        net.save_graph(path)
    
    # Load HTML into Streamlit
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Delete the temporary file
    os.unlink(path)
    
    # Return the HTML for display
    return html

def get_download_link(content, filename, text):
    """Generate a link to download content as a file"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def parse_text_to_json(text_data):
    """Parse text data into JSON format for our knowledge graph"""
    lines = text_data.strip().split('\n')
    nodes = []
    relationships = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # Try to detect if this is a relationship line
        if '"' in line and len(line.split('"')) >= 3:
            # This looks like a relationship
            parts = line.split('"')
            if len(parts) >= 3:
                # Format is expected to be like:
                # 0"id='School Board' type='Organization' properties={}"1"APPROVES"2"id='Expenditures' type='Concept' properties={}"
                source = parts[0]
                rel_type = parts[1]
                target = parts[2]
                
                relationships.append([source, rel_type, target])
        else:
            # Assume this is a node
            nodes.append(line)
    
    return {"nodes": nodes, "relationships": relationships}

def main():
    st.set_page_config(
        page_title="POWL Model Builder",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("POWL Model Builder for Legal Knowledge Graphs")
    st.write("""
    This application helps you convert legal knowledge graphs into POWL (Partially Ordered Workflow Language) models.
    Upload a JSON file containing your knowledge graph or paste your graph data directly.
    """)
    
    # Add tabs for different input methods
    input_tab1, input_tab2, input_tab3 = st.tabs(["Upload JSON", "Paste Text", "Sample Data"])
    
    json_data = None
    
    with input_tab1:
        # File upload
        uploaded_file = st.file_uploader("Upload Knowledge Graph JSON", type=["json"])
        
        if uploaded_file is not None:
            try:
                # Load and parse the JSON data
                json_data = json.load(uploaded_file)
                st.success("JSON file loaded successfully")
            except Exception as e:
                st.error(f"Error loading JSON: {str(e)}")
    
    with input_tab2:
        # Text input
        text_data = st.text_area("Paste Knowledge Graph Data", 
                                height=200, 
                                placeholder="""0"id='County Of Henrico, Virginia' type='Code' properties={}"1"DEFINED_AS"2"id='Code Of Virginia' type='Code' properties={}"
0"id='School Board' type='Organization' properties={}"1"APPROVES"2"id='Expenditures' type='Concept' properties={}"
""")
        
        if text_data:
            if st.button("Process Text Data"):
                try:
                    json_data = parse_text_to_json(text_data)
                    st.success("Text data processed successfully")
                except Exception as e:
                    st.error(f"Error processing text data: {str(e)}")
    
    with input_tab3:
        # Sample data
        if st.button("Use Sample Data"):
            json_data = {
                "nodes": [
                    "id='School Board' type='Organization' properties={}",
                    "id='Superintendent' type='Organization' properties={}",
                    "id='Budget' type='Document' properties={}",
                    "id='Expenditures' type='Concept' properties={}",
                    "id='Annual Report' type='Document' properties={}",
                    "id='County Of Henrico, Virginia' type='Code' properties={}",
                    "id='Code Of Virginia' type='Code' properties={}"
                ],
                "relationships": [
                    ["id='School Board' type='Organization' properties={}", "APPROVES", "id='Expenditures' type='Concept' properties={}"],
                    ["id='School Board' type='Organization' properties={}", "REVIEWS", "id='Budget' type='Document' properties={}"],
                    ["id='Superintendent' type='Organization' properties={}", "PREPARES", "id='Budget' type='Document' properties={}"],
                    ["id='Superintendent' type='Organization' properties={}", "SUBMITS", "id='Annual Report' type='Document' properties={}"],
                    ["id='County Of Henrico, Virginia' type='Code' properties={}", "DEFINED_AS", "id='Code Of Virginia' type='Code' properties={}"]
                ]
            }
            st.success("Sample data loaded")
    
    if json_data is not None:
        try:
            # Show raw data in expandable section
            with st.expander("Raw Data"):
                st.json(json_data)
            
            # Process the knowledge graph
            kg = KnowledgeGraph(json_data)
            
            # Show basic graph statistics
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Nodes", len(kg.graph.nodes()))
            col2.metric("Relationships", len(kg.graph.edges()))
            
            # Convert to POWL model
            powl_model = kg.to_powl()
            col3.metric("Tasks Identified", len(powl_model.tasks))
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["POWL Model", "Petri Net Visualization", "Download"])
            
            with tab1:
                st.header("POWL Model")
                
                # Display tasks
                st.subheader("Tasks/Actors")
                if powl_model.tasks:
                    task_data = []
                    for task in powl_model.tasks:
                        node_attrs = kg.graph.nodes[task]
                        task_data.append({
                            "Task ID": task,
                            "Type": node_attrs.get("type", "")
                        })
                    
                    tasks_df = pd.DataFrame(task_data)
                    st.dataframe(tasks_df)
                else:
                    st.write("No tasks identified.")
                
                # Display relationships table
                st.subheader("Relationships")
                relationships_data = []
                
                # Add control flow relationships
                for source, target in powl_model.control_flow:
                    # Find the edge data
                    edge_data = kg.graph.get_edge_data(source, target)
                    rel_type = edge_data.get("type", "") if edge_data else ""
                    
                    relationships_data.append({
                        "Source": source,
                        "Relationship": rel_type,
                        "Target": target,
                        "Flow Type": "Control Flow"
                    })
                
                # Add data flow relationships
                for source, target in powl_model.data_flow:
                    # Find the edge data
                    edge_data = kg.graph.get_edge_data(source, target)
                    rel_type = edge_data.get("type", "") if edge_data else ""
                    
                    relationships_data.append({
                        "Source": source,
                        "Relationship": rel_type,
                        "Target": target,
                        "Flow Type": "Data Flow"
                    })
                
                if relationships_data:
                    rel_df = pd.DataFrame(relationships_data)
                    st.dataframe(rel_df)
                else:
                    st.write("No relationships found.")
                
                # Display POWL model as graph
                st.subheader("POWL Model Visualization")
                
                # Add options for visualization
                col1, col2 = st.columns(2)
                with col1:
                    height = st.slider("Visualization Height", 400, 1200, 600, 100, key="powl_height")
                with col2:
                    physics = st.checkbox("Enable Physics Simulation", True, key="powl_physics")
                
                html = visualize_graph(kg.graph, height=height, physics_enabled=physics)
                components.html(html, height=height)
            
            with tab2:
                st.header("Petri Net Representation")
                
                # Convert to Petri Net
                petri_net = powl_model.to_petri_net()
                
                # Display Petri Net visualization
                st.subheader("Petri Net Visualization")
                
                # Add options for visualization
                col1, col2 = st.columns(2)
                with col1:
                    height = st.slider("Visualization Height", 400, 1200, 800, 100, key="petri_height")
                with col2:
                    physics = st.checkbox("Enable Physics Simulation", True, key="petri_physics")
                
                html = visualize_graph(petri_net, height=height, physics_enabled=physics)
                components.html(html, height=height)
                
                # Display explanation
                with st.expander("About Petri Nets"):
                    st.write("""
                    **Petri Net Representation**
                    
                    In this visualization:
                    - **Purple circles** represent places (conditions or states)
                    - **Blue boxes** represent transitions (tasks or activities)
                    - Each task has an input place (precondition) and output place (postcondition)
                    - Arrows show the flow between places and transitions
                    
                    Petri nets provide formal semantics for workflow processes, allowing analysis of properties like reachability, deadlocks, and concurrency.
                    """)
            
            with tab3:
                st.header("Download POWL Model")
                
                # Generate JSON for download
                powl_json = powl_model.to_json()
                
                # Create download link
                st.markdown(
                    get_download_link(powl_json, "powl_model.json", "Download POWL Model as JSON"),
                    unsafe_allow_html=True
                )
                
                # Display JSON preview
                with st.expander("Preview JSON"):
                    st.json(json.loads(powl_json))
                
                # Option to download Petri Net
                st.subheader("Download Petri Net Model")
                
                # Convert Petri Net to JSON
                petri_net_data = nx.node_link_data(powl_model.to_petri_net())
                petri_net_json = json.dumps(petri_net_data, indent=2)
                
                # Create download link for Petri Net
                st.markdown(
                    get_download_link(petri_net_json, "petri_net_model.json", "Download Petri Net Model as JSON"),
                    unsafe_allow_html=True
                )
                
                # Generate DOT format for Graphviz
                try:
                    from networkx.drawing.nx_agraph import write_dot
                    import io
                    
                    dot_buffer = io.StringIO()
                    write_dot(petri_net, dot_buffer)
                    dot_data = dot_buffer.getvalue()
                    
                    st.markdown(
                        get_download_link(dot_data, "petri_net.dot", "Download Petri Net as DOT file (for Graphviz)"),
                        unsafe_allow_html=True
                    )
                except ImportError:
                    st.info("For DOT file export, install pygraphviz package")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)
    
    # Add sidebar information
    with st.sidebar:
        st.header("About POWL Models")
        st.write("""
        POWL (Partially Ordered Workflow Language) is a formal language for modeling workflows 
        with partial ordering of activities.
        
        It's particularly useful for representing legal processes where activities may have 
        complex dependencies and relationships.
        """)
        
        st.subheader("Knowledge Graph Format")
        st.write("""
        This app expects knowledge graph data with nodes and relationships:
        
        **Node format:**
        ```
        id='Entity Name' type='Entity Type' properties={}
        ```
        
        **Relationship format:**
        ```
        id='Source Entity' type='Type' properties={}RELATIONSHIP_TYPEid='Target Entity' type='Type' properties={}
        ```
        
        For example:
        ```
        id='School Board' type='Organization' properties={}APPROVES id='Expenditures' type='Concept' properties={}
        ```
        """)
        
        st.subheader("References")
        st.markdown("[POWL Paper](https://sebastiaanvanzelst.com/wp-content/uploads/2023/08/paper_6723.pdf)")
        st.markdown("This app implements concepts from the paper to model legal workflows using POWL and Petri Nets.")

if __name__ == "__main__":
    main()