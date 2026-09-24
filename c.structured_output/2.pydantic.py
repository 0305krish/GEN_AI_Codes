from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import date

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash"
)

#schema(name, poduct, age, email, phone number, price, discount, rating, date)

class Customer(BaseModel):
    name : str
    product : str = Field(description = "Name of the product")
    age : Optional[int] = None
    email : str
    phone_no : str
    price : float
    discount : Optional[float] = Field(gt = 0, lt = 100)
    rating : Optional[float] = Field(gt = 0, ls = 5)
    date : date

structured_output = model.with_structured_output(Customer)

response = structured_output.invoke("""Hi, my name is Rahul Sharma. I recently purchased a Sony WH-CH720N headphone from your store for ₹8,500. I’m 20 years old and my email is rahul.sharma@gmail.com. You can contact me at 9876543210. I received a 10% discount on the product, and I would rate my overall experience 4.5 out of 5. The purchase was made on September 21, 2026.""")

print(response)

