
from backend import config
import os
import ollama


class DocumentGenerator:
    """
    A dynamic document generator using Ollama and user-defined templates.
    Automatically organizes files into folders by document type.
    """

    def __init__(self, model_name=config.MODEL, temperature=0.7, num_predict=10000):
        self.model = model_name
        self.temperature = temperature
        self.num_predict = num_predict

    # ------------------------------
    # Core Ollama Integration
    # ------------------------------
    def generate_document(self, prompt: str) -> str:
        """Generate text from a prompt using Ollama."""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": self.num_predict}
        )
        return response["response"]

    # ------------------------------
    # Template Utilities
    # ------------------------------
    @staticmethod
    def fill_template(query: str, context: dict) -> str:
        """Fill a prompt template with variables from a dictionary."""
        try:
            return query.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing placeholder in context: {e}")

    @staticmethod
    def safe_filename(name: str) -> str:
        """Convert a name into a safe filename."""
        return name.replace(" ", "_").replace("/", "_").replace("-", "_")

    # ------------------------------
    # Document Generation
    # ------------------------------
    def process_documents(
        self,
        template: str,
        data_list: list,
        doc_type: str = "generic",
        base_output_dir: str = "./generated_docs",
        header_template: str = None,
    ):
        """Generate multiple documents and save them under a type-specific folder."""
        output_dir = os.path.join(base_output_dir, doc_type.lower())
        os.makedirs(output_dir, exist_ok=True)

        for i, item in enumerate(data_list, 1):
            # Determine a name for the document (employee_name, product_type, etc.)
            name_key = (
                item.get("employee_name")
                or item.get("product_type")
                or item.get("title")
                or item.get("name")
                or f"Document_{i}"
            )

            print(f"[{i}/{len(data_list)}] Generating {doc_type} file for {name_key}")

            # Fill in template
            prompt = self.fill_template(template, item)
            content = self.generate_document(prompt)

            # Create header if provided
            header = ""
            if header_template:
                header = header_template.format(doc_number=1000 + i)

            # Save to markdown file
            filename = os.path.join(output_dir, f"{self.safe_filename(name_key)}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(header + content)

            print(f"✓ Saved: {filename}")

        print(f"\n✓ Done! All {doc_type} files saved in: {output_dir}\n")

    # ------------------------------
    # Convenience Entry Point
    # ------------------------------
    def run(
        self,
        template: str,
        data_list: list,
        doc_type: str,
        base_output_dir: str = "./generated_docs",
        header_template: str = """---
        Company: Oilwell Corporation
        Document Number: OW-DOC-{doc_number}
        Date: 2025-10-07
        Classification: Internal Use Only
        ---
        
        """,
        ):
        """Simplified entrypoint for document generation."""
        self.process_documents(
            template=template,
            data_list=data_list,
            doc_type=doc_type,
            base_output_dir=base_output_dir,
            header_template=header_template,
        )


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    generator = DocumentGenerator()

    mode = "contract"  # can be "product", "contract", etc.

    if mode == "product":
        template = """Generate product documentation for Oilwell Corporation.

        Product Type: {product_type}
        
        Format as markdown with sections:
        # [Product Name]
        ## Overview
        [2-3 paragraphs]
        ## Features
        - [Feature 1]
        - [Feature 2]
        - [Feature 3]
        ## Applications
        [List applications]
        """
        data = [
            {"product_type": "Centrifugal Pump - Multistage"},
            {"product_type": "Pressure Sensor - Digital Transmitter"},
            {"product_type": "Leak Detection and Repair - Tools"},
            {"product_type": "Wellhead Control Valve - High Pressure Gate Valve"},
            {"product_type": "Downhole Drilling Tool - Rotary Steerable System"},
        ]
        generator.run(template, data, doc_type="products")

    elif mode == "employee":
        template = """Generate an employee profile for Oilwell Corporation.

        Employee Name: {employee_name}
        Department: {department}
        Position: {position}
        Hire Date: {hire_date}
        
        Format as markdown with sections:
        # Employee Profile: {employee_name}
        ## Summary
        [Brief summary]
        ## Responsibilities
        - [Responsibility 1]
        - [Responsibility 2]
        - [Responsibility 3]
        ## Skills
        - [Skill 1]
        - [Skill 2]
        - [Skill 3]
        """
        data = [
            {
                "employee_name": "Nene Smith",
                "department": "c-suite",
                "position": "CEO",
                "hire_date": "2022-07-01",
            },
            {
                "employee_name": "Chi Doe",
                "department": "c-suite",
                "position": "CTO",
                "hire_date": "2018-03-16",
            },
            {
                "employee_name": "John Doe",
                "department": "Engineering",
                "position": "Drilling Engineer",
                "hire_date": "2018-03-15",
            },
            {
                "employee_name": "Jane Smith",
                "department": "Safety",
                "position": "Safety Officer",
                "hire_date": "2020-07-01",
            },
            {"employee_name": "Carlos Martinez",
             "department": "Operations",
             "position": "Field Supervisor",
             "hire_date": "2016-09-10"
             },
        ]
        generator.run(template, data, doc_type="employees")

    elif mode == "contract":
        template = """Generate a business contract summary for Oilwell Corporation.

        Contract Title: {title}
        Parties: {parties}
        Effective Date: {effective_date}
        Term: {term}
        
        Format as markdown with sections:
        # Contract: {title}
        ## Overview
        [Summary of the contract]
        ## Key Terms
        [List of 5 major clauses]
        ## Obligations
        [List obligations of both parties]
        ## Renewal & Termination
        [Summarize renewal and termination terms]
        """
        data = [
            {
                "title": "Supply Agreement - ABC Industrial Co.",
                "parties": "Oilwell Corporation and ABC Industrial Co.",
                "effective_date": "2024-01-01",
                "term": "2 years",
            },
            {
                "title": "Maintenance Contract - North Drilling Services",
                "parties": "Oilwell Corporation and North Drilling Services Ltd.",
                "effective_date": "2023-06-15",
                "term": "3 years",
            },
            {
                "title": "Logistics and Transportation Agreement - TransMove Logistics",
                "parties": "Oilwell Corporation and TransMove Logistics Ltd.",
                "effective_date": "2024-05-10",
                "term": "1 year, renewable",
            },
            {
                "title": "Consulting Services Agreement - PetroConsult Energy Advisors",
                "parties": "Oilwell Corporation and PetroConsult Energy Advisors Inc.",
                "effective_date": "2025-01-01",
                "term": "18 months",
            },
            {
                "title": "Software Licensing Agreement - TechWave Solutions",
                "parties": "Oilwell Corporation and TechWave Solutions LLC",
                "effective_date": "2024-09-01",
                "term": "3 years",
            },
        ]
        generator.run(template, data, doc_type="contracts")

    elif mode == "policy":
        template = """Generate a company policy document for Oilwell Corporation.

        Policy Title: {title}
        Department Responsible: {department}
        Effective Date: {effective_date}
        Review Cycle: {review_cycle}

        Format as markdown with these sections:

        # Policy: {title}

        ## Purpose
        [Explain why this policy exists and its objectives.]

        ## Scope
        [Define who and what this policy applies to.]

        ## Policy Statement
        [State the key rules, standards, or expectations clearly.]

        ## Procedures
        [List the specific procedures or steps employees should follow.]

        ## Responsibilities
        [Define the roles and responsibilities of employees, managers, and departments.]

        ## Compliance
        [Outline how compliance will be monitored and enforced.]

        ## Review & Revision
        [Describe the review cycle and update procedures.]
        """

        data = [
            {
                "title": "Workplace Health and Safety Policy",
                "department": "Health, Safety, and Environment (HSE)",
                "effective_date": "2025-01-01",
                "review_cycle": "Annual",
            },
            {
                "title": "Code of Conduct and Ethics Policy",
                "department": "Human Resources / Compliance",
                "effective_date": "2025-01-01",
                "review_cycle": "Every 2 years",
            },
            {
                "title": "Data Protection and Privacy Policy",
                "department": "Information Technology (IT)",
                "effective_date": "2025-02-01",
                "review_cycle": "Annual",
            },
            {
                "title": "Environmental Sustainability Policy",
                "department": "Health, Safety, and Environment (HSE)",
                "effective_date": "2025-03-01",
                "review_cycle": "Every 3 years",
            },
            {
                "title": "Equal Employment Opportunity Policy",
                "department": "Human Resources",
                "effective_date": "2025-01-15",
                "review_cycle": "Every 2 years",
            },
        ]

        generator.run(template, data, doc_type="policies")


