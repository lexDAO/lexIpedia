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
import time
import gc
from tqdm import tqdm
import concurrent.futures
import threading

# Add OpenAI support
import openai
from typing import List, Dict, Any, Optional, Tuple

class OpenAIProcessor:
    """Process knowledge graphs using OpenAI for large datasets"""
    def __init__(self, api_key=None):
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Try to use environment variable
            self.client = openai.OpenAI()  # Will use OPENAI_API_KEY env var if set
        
    def extract_relationships(self, data: Dict[str, Any], chunk_size=50) -> Tuple[List[str], List[List]]:
        """Extract relationships from large datasets using OpenAI"""
        st.info("Using OpenAI to process large dataset...")
        
        # Get nodes and store them
        nodes = data.get("nodes", [])
        unique_nodes = set()
        
        # Extract nodes
        for node in tqdm(nodes, desc="Processing nodes"):
            if isinstance(node, dict) and "id" in node:
                unique_nodes.add(node["id"])
            elif isinstance(node, str):
                match = re.search(r"id=['\"]([^'\"]+)['\"]", node)
                if match:
                    unique_nodes.add(match.group(1))
        
        st.info(f"Found {len(unique_nodes)} unique nodes")
        
        # Get relationships
        relationships = data.get("relationships", [])
        processed_relationships = []
        
        # Process relationships in chunks to avoid timeout
        chunks = [relationships[i:i+chunk_size] for i in range(0, len(relationships), chunk_size)]
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            # Update progress
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            
            # Process this chunk
            try:
                prompt = self._create_relationship_prompt(chunk)
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a legal knowledge graph processor. Extract source, relationship type, and target from each relationship."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
                # Parse the response
                result = response.choices[0].message.content
                chunk_relationships = self._parse_openai_response(result)
                processed_relationships.extend(chunk_relationships)
                
                # Give the API a short break to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                st.warning(f"Error processing chunk {i+1}: {str(e)}")
        
        # Convert nodes to the expected format
        formatted_nodes = []
        for node_id in unique_nodes:
            formatted_nodes.append(f"id='{node_id}' type='Entity' properties={{}}") 
            
        return formatted_nodes, processed_relationships
    
    def _create_relationship_prompt(self, relationships: List) -> str:
        """Create a prompt for relationship extraction"""
        prompt = """
        I have relationship data from a legal knowledge graph. For each relationship, extract:
        1. The source entity ID
        2. The relationship type
        3. The target entity ID
        
        Return results in CSV format with three columns: source,relationship,target
        
        Here are the relationships:
        """
        
        for rel in relationships:
            if isinstance(rel, list) and len(rel) >= 3:
                prompt += f"\n- {rel}"
            elif isinstance(rel, str):
                prompt += f"\n- {rel}"
            else:
                prompt += f"\n- {str(rel)}"
                
        return prompt
    
    def _parse_openai_response(self, response: str) -> List[List]:
        """Parse OpenAI response into relationships"""
        relationships = []
        
        # Skip header line if present
        lines = response.strip().split('\n')
        start_line = 0
        if "source" in lines[0].lower() and "relationship" in lines[0].lower():
            start_line = 1
            
        for line in lines[start_line:]:
            if not line.strip():
                continue
                
            # Handle CSV format with or without quotes
            parts = []
            in_quotes = False
            current_part = ""
            
            for char in line:
                if char == '"' or char == "'":
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
                    
            if current_part:
                parts.append(current_part.strip())
                
            if len(parts) >= 3:
                source_id = parts[0]
                rel_type = parts[1]
                target_id = parts[2]
                
                # Format for our app
                source = f"id='{source_id}' type='Entity' properties={{}}"
                target = f"id='{target_id}' type='Entity' properties={{}}"
                
                relationships.append([source, rel_type, target])
                
        return relationships

class KnowledgeGraph:
    def __init__(self, json_data, max_nodes=None, max_relationships=None):
        self.data = json_data
        self.graph = nx.DiGraph()
        self.max_nodes = max_nodes
        self.max_relationships = max_relationships
        self.parse_knowledge_graph()
    
    def parse_knowledge_graph(self):
        """Parse JSON knowledge graph into a NetworkX graph"""
        # Get nodes and relationships from the data
        nodes = self.data.get("nodes", [])
        relationships = self.data.get("relationships", [])
        
        # Apply limits if specified
        if self.max_nodes:
            nodes = nodes[:self.max_nodes]
        if self.max_relationships:
            relationships = relationships[:self.max_relationships]
        
        # Create progress bars
        node_progress = st.progress(0)
        node_status = st.empty()
        node_status.text("Processing nodes...")
        
        # Add all nodes first - with batch processing
        batches = [nodes[i:i+1000] for i in range(0, len(nodes), 1000)]
        
        for i, batch in enumerate(batches):
            nodes_processed = 0
            
            for node in batch:
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
                    nodes_processed += 1
                    
                except Exception as e:
                    # Just continue on error
                    pass
            
            # Update progress
            progress = (i + 1) / len(batches)
            node_progress.progress(progress)
            node_status.text(f"Processing nodes: {i*1000 + nodes_processed}/{len(nodes)}")
        
        # Clear the progress bars
        node_status.empty()
        node_progress.empty()
        
        # Now process relationships
        rel_progress = st.progress(0)
        rel_status = st.empty()
        rel_status.text("Processing relationships...")
        
        # Process relationships in batches
        rel_batches = [relationships[i:i+1000] for i in range(0, len(relationships), 1000)]
        
        for i, batch in enumerate(rel_batches):
            rels_processed = 0
            
            for rel in batch:
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
                        
                    elif isinstance(rel, str):
                        # Try to extract from a string format
                        parts = rel.split('"')
                        if len(parts) >= 5:
                            source_info = parts[0]
                            relationship_type = parts[1]
                            target_info = parts[2]
                            
                            source_id = self._extract_value(source_info, "id")
                            target_id = self._extract_value(target_info, "id")
                        else:
                            # Try alternate parsing for strings
                            relation_match = re.search(r"(.*?)([A-Z_]+)(.*)", rel)
                            if relation_match:
                                source_info = relation_match.group(1)
                                relationship_type = relation_match.group(2)
                                target_info = relation_match.group(3)
                                
                                source_id = self._extract_value(source_info, "id")
                                target_id = self._extract_value(target_info, "id")
                            else:
                                continue
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
                        
                        # Add the edge with relationship type (ensure it's a string)
                        self.graph.add_edge(source_id, target_id, type=str(relationship_type))
                        rels_processed += 1
                
                except Exception as e:
                    # Just continue on error
                    pass
            
            # Update progress
            progress = (i + 1) / len(rel_batches)
            rel_progress.progress(progress)
            rel_status.text(f"Processing relationships: {i*1000 + rels_processed}/{len(relationships)}")
            
            # Force garbage collection periodically
            if i % 5 == 0:
                gc.collect()
        
        # Clear progress indicators
        rel_status.empty()
        rel_progress.empty()
        
        # Summary
        st.success(f"Loaded {len(self.graph.nodes())} nodes and {len(self.graph.edges())} relationships")
    
    def _extract_value(self, text, key):
        """Extract a value from a string like id='X' type='Y'"""
        if not isinstance(text, str):
            return ""
            
        pattern = f"{key}=['\"](.*?)['\"]"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return ""
    
    def to_powl(self):
        """Convert knowledge graph to POWL model"""
        powl_model = POWLModel(self.graph)
        return powl_model
    
    def extract_subgraph(self, max_nodes=100):
        """Extract a manageable subgraph for visualization"""
        if len(self.graph) <= max_nodes:
            return self.graph
            
        # Get central nodes (nodes with highest degree)
        central_nodes = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes//2]
        central_node_ids = [node[0] for node in central_nodes]
        
        # Create subgraph
        subgraph = nx.DiGraph()
        
        # Add central nodes
        for node_id in central_node_ids:
            if node_id in self.graph:
                subgraph.add_node(node_id, **self.graph.nodes[node_id])
        
        # Add some neighbors to fill out the graph
        remaining_slots = max_nodes - len(central_node_ids)
        neighbors = set()
        
        for node_id in central_node_ids:
            neighbors.update(self.graph.successors(node_id))
            neighbors.update(self.graph.predecessors(node_id))
        
        # Remove nodes already in central_nodes
        neighbors = [n for n in neighbors if n not in central_node_ids]
        
        # Add remaining neighbors up to max_nodes
        for node_id in neighbors[:remaining_slots]:
            if node_id in self.graph:
                subgraph.add_node(node_id, **self.graph.nodes[node_id])
        
        # Add edges between nodes in the subgraph
        for source, target in self.graph.edges():
            if source in subgraph and target in subgraph:
                edge_data = self.graph.get_edge_data(source, target)
                subgraph.add_edge(source, target, **edge_data)
        
        return subgraph

class POWLModel:
    def __init__(self, graph):
        self.graph = graph
        self.tasks = []
        self.control_flow = []
        self.data_flow = []
        
        # Map relationship types to flow types - extended for legal domain
        self.control_flow_types = [
            # Standard workflow relationships
            "FOLLOWS", "PRECEDES", "NEXT", "SEQUENCE", 
            "REQUIRES", "DEPENDS_ON", "TRIGGERS",
            
            # Legal authority relationships
            "APPROVES", "AUTHORIZES", "DECIDES", "GRANTS",
            "PERMITS", "ALLOWS", "ENFORCES", "REGULATES",
            "SUPERVISES", "OVERSEES", "DIRECTS", "MANAGES",
            "ADMINISTERS", "GOVERNS", "RULES", "CONTROLS",
            
            # Procedural relationships
            "INITIATES", "TERMINATES", "APPEALS", "REVIEWS",
            "ADJUDICATES", "ARBITRATES", "MEDIATES", "RESOLVES",
            "DETERMINES", "EVALUATES", "ASSESSES", "VALIDATES",
            "CONFIRMS", "VERIFIES", "CERTIFIES", "APPROVES"
        ]
        
        self.data_flow_types = [
            # Standard data relationships
            "USES", "PRODUCES", "CONSUMES", "CREATES", "DATA",
            "INPUTS", "OUTPUTS", "READS", "WRITES", "MODIFIES",
            
            # Legal document relationships
            "DEFINED_AS", "REFERENCES", "CITES", "AMENDS",
            "SUPERSEDES", "REPEALS", "ENACTS", "CODIFIES",
            "INTERPRETS", "CLARIFIES", "EXPLAINS", "DEFINES",
            "SPECIFIES", "DETAILS", "OUTLINES", "CONTAINS",
            "INCLUDES", "EXCLUDES", "EXEMPTS", "DESCRIBES"
        ]
        
        self.process_graph()
    
    def process_graph(self):
        """Process the graph to identify tasks, control flow, and data flow"""
        # Create status indicator
        status = st.empty()
        status.text("Processing graph for POWL model...")
        
        # Identify tasks (activities)
        # Extended list of task/actor types for legal domain
        task_types = [
            # Standard workflow participants
            "ACTIVITY", "TASK", "PROCESS", "STEP", "ACTION", 
            
            # Legal entities/actors
            "ORGANIZATION", "DEPARTMENT", "ENTITY", "BOARD", 
            "COMMITTEE", "AUTHORITY", "AGENCY", "COMMISSION",
            "BUREAU", "DIVISION", "OFFICE", "BRANCH", "COUNCIL",
            "COURT", "TRIBUNAL", "PANEL", "BODY", "OFFICIAL",
            "OFFICER", "DIRECTOR", "ADMINISTRATOR", "JUDGE",
            "MAGISTRATE", "CLERK", "SECRETARY", "ATTORNEY",
            "REPRESENTATIVE", "AGENT", "PERSON"
        ]
        
        # Process nodes with progress tracking
        nodes_processed = 0
        total_nodes = len(self.graph.nodes())
        
        for node, attrs in self.graph.nodes(data=True):
            nodes_processed += 1
            if nodes_processed % 1000 == 0:
                status.text(f"Processing nodes: {nodes_processed}/{total_nodes}")
                
            node_type = attrs.get("type", "").upper()
            if any(task_type in node_type for task_type in task_types):
                self.tasks.append(node)
        
        # If no explicit task types are found, use heuristics
        if len(self.tasks) < 5:  # Very few tasks found
            status.text("Few tasks found, identifying potential actors based on relationships...")
            
            # Find nodes that have significant outgoing relationships (likely actors)
            for node in self.graph.nodes():
                if self.graph.out_degree(node) >= 2:  # Has multiple outgoing edges
                    self.tasks.append(node)
            
            # Limit to reasonable number of tasks
            if len(self.tasks) > 100:
                # Keep only the most connected ones
                self.tasks = sorted(self.tasks, key=lambda x: self.graph.degree(x), reverse=True)[:100]
        
        status.text(f"Identified {len(self.tasks)} tasks/actors. Processing relationships...")
        
        # Set to track processed edges to avoid duplicates
        processed_edges = set()
        edges_processed = 0
        total_edges = len(self.graph.edges())
        
        # Identify control flow and data flow relationships
        for source, target, attrs in self.graph.edges(data=True):
            edges_processed += 1
            if edges_processed % 1000 == 0:
                status.text(f"Processing relationships: {edges_processed}/{total_edges}")
                
            # Skip if already processed
            edge_key = (source, target)
            if edge_key in processed_edges:
                continue
                
            processed_edges.add(edge_key)
            
            edge_type = str(attrs.get("type", "")).upper()
            
            if edge_type in self.control_flow_types:
                self.control_flow.append((source, target))
            elif edge_type in self.data_flow_types:
                self.data_flow.append((source, target))
            else:
                # If relationship type is not recognized, use heuristics
                if source in self.tasks and target in self.tasks:
                    # Both nodes are tasks/actors - likely control flow
                    self.control_flow.append((source, target))
                else:
                    # One node is not a task - likely data flow
                    self.data_flow.append((source, target))
        
        status.empty()
    
    def to_petri_net(self, max_tasks=50):
        """Convert POWL model to Petri Net representation"""
        # Limit to manageable size
        task_subset = self.tasks[:max_tasks] if len(self.tasks) > max_tasks else self.tasks
        
        petri_net = nx.DiGraph()
        
        # Create places for each task
        for task in task_subset:
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
            if source in task_subset and target in task_subset:
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

def visualize_graph(graph, height=800, physics_enabled=True, node_limit=100):
    """Create interactive visualization of the graph"""
    # Create a pyvis network
    net = Network(height=f"{height}px", width="100%", notebook=True, directed=True)
    
    # Limit number of nodes for visualization
    if len(graph.nodes()) > node_limit:
        st.warning(f"Graph is too large to visualize completely. Showing {node_limit} key nodes.")
        
        # Get most connected nodes
        central_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:node_limit]
        nodes_to_include = [node[0] for node in central_nodes]
        
        # Create subgraph
        subgraph = graph.subgraph(nodes_to_include)
    else:
        subgraph = graph
    
    # Add nodes
    for node, attrs in subgraph.nodes(data=True):
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
        elif any(t in str(node_type).upper() for t in ["ACTIVITY", "TASK", "PROCESS", "STEP", "ACTION"]):
            color = "#fa4d56"  # Red for activities/tasks
            shape = "box"
        elif any(t in str(node_type).upper() for t in ["ORGANIZATION", "DEPARTMENT", "ENTITY", "BOARD", "COMMITTEE"]):
            color = "#08bdba"  # Teal for organizations
            shape = "diamond"
        elif any(t in str(node_type).upper() for t in ["DOCUMENT", "FORM", "REPORT", "CODE"]):
            color = "#33b1ff"  # Light blue for documents
            shape = "file"
        elif any(t in str(node_type).upper() for t in ["CONCEPT", "TERM", "DEFINITION"]):
            color = "#d2a106"  # Gold for concepts
            shape = "ellipse"
        else:
            color = "#8a3ffc"  # Default color
            shape = "ellipse"
        
        # Add the node with appropriate attributes
        net.add_node(node, label=display_label, title=f"{node_type}: {label}", 
                    color=color, shape=shape)
    
    # Add edges
    for source, target, attrs in subgraph.edges(data=True):
        edge_type = attrs.get("type", "")
        title = edge_type if edge_type else "connects"
        
        # Choose edge color based on relationship type
        if str(edge_type).upper() in ["FOLLOWS", "PRECEDES", "NEXT", "SEQUENCE", "APPROVES"]:
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
    
    try:
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
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return f"<div>Error generating visualization: {str(e)}</div>"

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

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    return {
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
    
    # Add configuration options
    with st.expander("⚙️ Configuration Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            limit_nodes = st.checkbox("Limit nodes for better performance", True)
            if limit_nodes:
                max_nodes = st.number_input("Maximum nodes to process", min_value=100, max_value=10000, value=1000)
            else:
                max_nodes = None
            
            limit_relationships = st.checkbox("Limit relationships for better performance", True)
            if limit_relationships:
                max_relationships = st.number_input("Maximum relationships to process", min_value=100, max_value=20000, value=2000)
            else:
                max_relationships = None
        
        with col2:
            visualization_height = st.slider("Visualization height (px)", min_value=400, max_value=1200, value=800)
            physics_enabled = st.checkbox("Enable physics simulation", True)
            node_vis_limit = st.number_input("Max nodes to visualize", min_value=20, max_value=500, value=100)
            
            # OpenAI API integration
            use_openai = st.checkbox("Enable OpenAI for large datasets", False)
            if use_openai:
                openai_api_key = st.text_input("OpenAI API Key", type="password")
                openai_chunk_size = st.slider("Chunk size for processing", min_value=10, max_value=100, value=50)
            else:
                openai_api_key = None
                openai_chunk_size = 50
    
    # Input methods tabs
    input_method = st.radio("Select input method:", ["Upload JSON", "Paste Text", "Use Sample Data"])
    
    graph_data = None
    
    if input_method == "Upload JSON":
        uploaded_file = st.file_uploader("Upload knowledge graph JSON file", type=['json'])
        if uploaded_file is not None:
            try:
                graph_data = json.load(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif input_method == "Paste Text":
        text_data = st.text_area("Paste knowledge graph data (nodes and relationships)", height=200)
        if st.button("Process Text Data"):
            if text_data:
                try:
                    graph_data = parse_text_to_json(text_data)
                    st.success("Text processed successfully!")
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
    
    else:  # Use Sample Data
        if st.button("Load Sample Data"):
            graph_data = load_sample_data()
            st.success("Sample data loaded!")
    
    # Process data if available
    if graph_data:
        # Use OpenAI for large datasets if enabled
        if use_openai and openai_api_key and (len(graph_data.get("nodes", [])) > 1000 or len(graph_data.get("relationships", [])) > 2000):
            try:
                processor = OpenAIProcessor(api_key=openai_api_key)
                nodes, relationships = processor.extract_relationships(graph_data, chunk_size=openai_chunk_size)
                graph_data = {"nodes": nodes, "relationships": relationships}
            except Exception as e:
                st.error(f"Error processing with OpenAI: {str(e)}")
        
        # Process knowledge graph
        with st.spinner("Processing knowledge graph..."):
            kg = KnowledgeGraph(graph_data, max_nodes=max_nodes, max_relationships=max_relationships)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "POWL Model", "Petri Net", "Export"])
            
            with tab1:
                st.header("Knowledge Graph Visualization")
                
                # Get subgraph for visualization
                viz_graph = kg.extract_subgraph(max_nodes=node_vis_limit)
                
                # Create visualization
                html = visualize_graph(viz_graph, height=visualization_height, 
                                     physics_enabled=physics_enabled,
                                     node_limit=node_vis_limit)
                
                # Display visualization
                components.html(html, height=visualization_height+50)
                
                # Display graph stats
                st.subheader("Graph Statistics")
                st.write(f"Total nodes: {len(kg.graph.nodes())}")
                st.write(f"Total relationships: {len(kg.graph.edges())}")
                
                # Top 10 nodes by degree
                st.subheader("Most Connected Nodes")
                top_nodes = sorted(kg.graph.degree(), key=lambda x: x[1], reverse=True)[:10]
                top_nodes_df = pd.DataFrame(top_nodes, columns=["Node", "Connections"])
                st.dataframe(top_nodes_df)
            
            with tab2:
                st.header("POWL Model")
                
                # Convert to POWL model
                powl_model = kg.to_powl()
                
                # Display POWL model details
                st.subheader("Tasks/Actors")
                st.write(f"Total tasks/actors identified: {len(powl_model.tasks)}")
                if len(powl_model.tasks) > 0:
                    tasks_df = pd.DataFrame(powl_model.tasks[:20], columns=["Task/Actor"])
                    st.dataframe(tasks_df)
                    if len(powl_model.tasks) > 20:
                        st.info(f"Showing 20 of {len(powl_model.tasks)} tasks/actors")
                
                st.subheader("Control Flow")
                st.write(f"Total control flow relationships: {len(powl_model.control_flow)}")
                if len(powl_model.control_flow) > 0:
                    control_flow_df = pd.DataFrame(powl_model.control_flow[:20], columns=["Source", "Target"])
                    st.dataframe(control_flow_df)
                    if len(powl_model.control_flow) > 20:
                        st.info(f"Showing 20 of {len(powl_model.control_flow)} control flow relationships")
                
                st.subheader("Data Flow")
                st.write(f"Total data flow relationships: {len(powl_model.data_flow)}")
                if len(powl_model.data_flow) > 0:
                    data_flow_df = pd.DataFrame(powl_model.data_flow[:20], columns=["Source", "Target"])
                    st.dataframe(data_flow_df)
                    if len(powl_model.data_flow) > 20:
                        st.info(f"Showing 20 of {len(powl_model.data_flow)} data flow relationships")
            
            with tab3:
                st.header("Petri Net Representation")
                
                # Convert to Petri Net
                petri_net = powl_model.to_petri_net(max_tasks=50)
                
                # Create visualization
                petri_html = visualize_graph(petri_net, height=visualization_height, 
                                          physics_enabled=physics_enabled,
                                          node_limit=node_vis_limit)
                
                # Display visualization
                components.html(petri_html, height=visualization_height+50)
                
                # Display Petri Net stats
                st.subheader("Petri Net Statistics")
                places = [n for n, attr in petri_net.nodes(data=True) if attr.get('type') == 'place']
                transitions = [n for n, attr in petri_net.nodes(data=True) if attr.get('type') == 'transition']
                
                st.write(f"Places: {len(places)}")
                st.write(f"Transitions: {len(transitions)}")
                st.write(f"Arcs: {len(petri_net.edges())}")
            
            with tab4:
                st.header("Export Options")
                
                # JSON export of POWL model
                powl_json = powl_model.to_json()
                st.subheader("POWL Model (JSON)")
                st.text_area("POWL JSON", powl_json, height=200)
                st.markdown(get_download_link(powl_json, "powl_model.json", "Download POWL Model as JSON"), unsafe_allow_html=True)
                
                # Export original graph as JSON
                graph_json = json.dumps({
                    "nodes": list(kg.graph.nodes()),
                    "relationships": list(kg.graph.edges(data=True))
                }, indent=2)
                
                st.subheader("Original Graph (JSON)")
                st.text_area("Graph JSON", graph_json, height=200)
                st.markdown(get_download_link(graph_json, "knowledge_graph.json", "Download Graph as JSON"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()