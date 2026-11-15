
def evaluate_hotel_result(hotel, expected_city, max_price):
    city_correct = hotel["city"] == expected_city
    price_correct = hotel["price"] <= max_price

    return {
        "city_correct": city_correct,
        "price_correct": price_correct,
        "output_correct": city_correct and price_correct
    }


hotel = {
    "name": "Hotel Sakura",
    "city": "Tokyo",
    # "price": 350
    "price": 180
}

result = evaluate_hotel_result(
    hotel=hotel,
    expected_city="Tokyo",
    max_price=200
)

print("\nTool Output Evaluation")
print("----------------------")
print("Hotel:", hotel)
print("Evaluation:", result)

# poetry run python lesson4_4_tool_output_evaluation.py
