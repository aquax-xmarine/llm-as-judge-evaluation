from langchain_core.documents import Document

# Sample knowledge base
documents = [
    Document(
        page_content="All customers are eligible for a full refund within 30 days of purchase. "
        "The item must be in its original packaging and unused condition. Refunds are "
        "processed to the original payment method within 5-7 business days.",
        metadata={"source": "return_policy.pdf", "section": "refund_eligibility"}
    ),
    Document(
        page_content="Exchanges can be requested within 45 days of purchase for items of "
        "equal or lesser value. Size exchanges on clothing are free of charge. "
        "For items of greater value, the customer pays the difference.",
        metadata={"source": "return_policy.pdf", "section": "exchanges"}
    ),
    Document(
        page_content="Electronics have a 15-day return window due to rapid depreciation. "
        "Opened software and digital downloads are non-refundable. Defective electronics "
        "can be returned within 90 days with proof of defect from an authorized service center.",
        metadata={"source": "return_policy.pdf", "section": "electronics"}
    ),
    Document(
        page_content="Shipping costs for returns are covered by the company for defective items. "
        "For non-defective returns, the customer is responsible for return shipping. "
        "Free return shipping labels are available for loyalty program members regardless of reason.",
        metadata={"source": "return_policy.pdf", "section": "shipping"}
    ),
    Document(
        page_content="Gift purchases can be returned with the gift receipt for store credit only. "
        "Without a gift receipt, returns are processed at the lowest sale price in the last 90 days. "
        "Gift cards and prepaid cards are non-refundable and cannot be exchanged.",
        metadata={"source": "return_policy.pdf", "section": "gifts"}
    ),
]



eval_questions = [
    # Straightforward questions (should be easy)
    "What is the refund window for regular purchases?",
    "Are exchanges free for clothing size changes?",

    # Questions requiring synthesis across chunks
    "What are my options if I received a defective laptop 60 days ago?",

    # Edge cases likely to cause hallucination
    "Can I get a refund for a digital download I purchased yesterday?",
    "What happens if I return a gift without the gift receipt?",

    # Questions where context might be incomplete
    "Do you offer refunds for international orders?",
    "Can I return an item I bought on sale?",

    # Adversarial or tricky questions
    "If I'm a loyalty member, do I get free return shipping even for electronics?",
]