from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal
from config.city_tier import tier_1_cities, tier_2_cities

class UserModel (BaseModel):
    age: Annotated[int, Field(..., description="age of the user",gt=0,lt=120)]
    weight: Annotated[float, Field(..., description="weight of the user",gt=0)]
    height:  Annotated[float, Field(..., description="height of the user in meter",gt=0,lt=2.5)]
    income_lpa: Annotated[float, Field(..., description="income of the user in LPA",gt=0)]
    smoker: Annotated[bool, Field(..., description="is the user a smoker or not", )]
    city: Annotated[str,Field(..., description="city of the user")]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job','business_owner', 'unemployed', 'private_job'], Field(..., description='Occupation of the user')]
       
    @property
    def bmi(self) -> float:
        return self.weight/(self.height**2)
    
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3

    @field_validator('city')
    @classmethod
    def normalize_city(cls, value:str)->str:
        return value.title().strip()
