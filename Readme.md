Lexipedia
=========

Introduction
------------

Lexipedia aims to simplify communicating complex business processes extracted from local, state, federal, and international legal codes. Our goal is to make starting and managing a business more accessible, particularly for small enterprises that often lack resources for expensive specialists.

### Motivation

Starting and managing a business is challenging, especially when it comes to:

-   Addressing and mitigating risks
-   Identifying opportunities such as grants and loans
-   Navigating complex financial instruments (insurance, retirement policies, health-related products)
-   Understanding and complying with various legal requirements

Smaller businesses often lose out on even basic benefits due to the complexity and cost of accessing this information. Lexipedia aims to level the playing field by providing clear, visual representations of these processes.
<img width="1026" alt="Screenshot 2024-08-16 at 3 40 14 PM" src="https://github.com/user-attachments/assets/9ba86e35-6890-4337-a954-5d1b8339cbc5">
The layers circled in red - BPMN markup, granular contracts, and discovery, are the primary focal points of Lexipedia
### Our Approach

Lexipedia uses LLM-assisted process mining to create Business Process Model and Notation (BPMN) diagrams. These diagrams simplify the understanding and upkeep of various business processes, making them more approachable for entrepreneurs and small business owners.

Instead of navigating through multiple forms and complex legal documents, users can refer to streamlined, visual representations of processes like starting a business in a specific location.

Project Components
------------------

### 1\. AI Models for Training

This repository contains models used for training AI to extract and process legal information. These models are crucial for:

-   Parsing legal documents (pile of law, ask abe, etc...)
-   Identifying key processes and requirements (kyc, kyb, etc...)
-   Generating BPMN diagrams (may extend to other models like POWL)

### 2\. Research Resources

We provide a collection of resources for researchers interested in:

-   LLM-assisted process mining
-   Legal text processing
-   Business process modeling
-   BPMN diagram generation

### 3\. Infrastructure Code

This section includes code for:

-   Working with BPMN models
-   Uploading final processes to GitHub
-   Setting up necessary environments for data processing and model training

### 4\. Examples
<img width="1406" alt="Screenshot 2024-08-14 at 2 58 53 PM" src="https://github.com/user-attachments/assets/25b4603f-3539-496d-a6d5-d438b4ba1ad3">
This is an example of a BPMN model based on https://www.charlottesville.gov/216/Starting-a-Small-Business
there are sample files in the bpmn-models directory

