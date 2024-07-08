import requests
from fuzzywuzzy import process

MAX_TRY = 5

def fetch_address_data(address):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def search_address(query, data):
    addresses = [result['ADDRESS'] for result in data['results']]
    match = process.extractOne(query, addresses)
    return match

def broaden_query(query):
    parts = query.split()
    if len(parts) > 1:
        return ' '.join(parts[:-1])
    else:
        return query

def find_best_match_address(query):
    original_query = query
    attempts = 0
    max_attempts = MAX_TRY

    while attempts < max_attempts:
        data = fetch_address_data(query)
        
        if data and data['found'] > 0:
            match = search_address(original_query, data)
            if match:
                best_match_address = match[0]
                confidence_score = match[1]
                print(f"Best match: {best_match_address}\nConfidence: {confidence_score}%")
                for result in data['results']:
                    if result['ADDRESS'] == best_match_address:
                        return result
            else:
                print("No match found in the current data")
        else:
            print(f"No data found for query: {query}")

        query = broaden_query(query)
        attempts += 1

    print("No matches.")
    return None

query = "138683"
best_match_result = find_best_match_address(query)

if best_match_result:
    #print(f"Best Match: {best_match_result}")
    print(f"Coords: {float(best_match_result['LATITUDE']):.4f}, {float(best_match_result['LONGITUDE']):.4f}")
else:
    print("No suitable match found")
