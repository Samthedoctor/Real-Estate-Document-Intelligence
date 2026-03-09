# reindex_clean.py
from document_system import HybridDocumentSystem

# Initialize
system = HybridDocumentSystem()
system.reset_database()


# Index single PDF
pdf_files = [
    "222-rajpur-brochure.pdf",
    "max-house-brochure (1).pdf",
    "max-towers-brochure (1).pdf"
]

system.add_documents(pdf_files)
stats = system.get_collection_stats()
print(stats['documents'])
print("\n✓ Clean generic indexing complete!")

