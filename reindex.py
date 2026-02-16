# reindex_clean.py
from document_system import HybridDocumentSystem

# Initialize
system = HybridDocumentSystem()
system.reset_database()

# Index single PDF
pdf_files = [
    "Environment-Clearance-of-Estate-128-I.pdf"
]

system.add_documents(pdf_files)
print("\n✓ Clean generic indexing complete!")

