from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

customer_schema = {
    "title": "Customer",
    "description": "Customer purchase information",
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "product": {
            "type": "string",
            "description": "Name of the product"
        },
        "age": {
            "type": ["integer", "null"]
        },
        "email": {
            "type": "string"
        },
        "phone_no": {
            "type": "string"
        },
        "price": {
            "type": "number"
        },
        "discount": {
            "type": ["number", "null"],
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 100
        },
        "rating": {
            "type": ["number", "null"],
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 5
        },
        "date": {
            "type": "string",
            "format": "date"
        }
    },
    "required": [
        "name",
        "product",
        "email",
        "phone_no",
        "price",
        "date"
    ]
}

structured_output = model.with_structured_output(customer_schema)

response = structured_output.invoke(
    """Hi, my name is Rahul Sharma. I recently purchased a Sony WH-CH720N
    headphone from your store for ₹8,500. I’m 20 years old and my email is
    rahul.sharma@gmail.com. You can contact me at 9876543210. I received a
    10% discount on the product, and I would rate my overall experience
    4.5 out of 5. The purchase was made on September 21, 2026."""
)

print(response)