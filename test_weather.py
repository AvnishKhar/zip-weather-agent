from pprint import pprint

from weather_service import get_weather, lookup_postal_code


print("Testing ZIP-code lookup...")
pprint(lookup_postal_code("45202", "US"))

print("\nTesting weather lookup...")
pprint(get_weather("45202", "US", "fahrenheit"))