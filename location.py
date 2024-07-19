import requests

MAX_TRY = 5

def fetch_address_data(address):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def similarity_score(s1, s2):
    lev_distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100
    return (1 - lev_distance / max_len) * 100

def levenshteinSelect(query, choices, processor=None, score_cutoff=0):
    if processor is None:
        processor = lambda x: x.lower().strip()

    processed_query = processor(query)
    best_match = None
    highest_score = score_cutoff

    for choice in choices:
        processed_choice = processor(choice)
        score = similarity_score(processed_query, processed_choice)
        if score > highest_score:
            highest_score = score
            best_match = (choice, score)

    return best_match

def search_address(query, data):
    addresses = [result['ADDRESS'] for result in data['results']]
    match = levenshteinSelect(query, addresses)
    return match

def broaden_query(query, attempt):
    parts = query.split()
    if len(parts) > 1:
        if attempt == 1:
            return ' '.join(parts[:-1])  # Remove last word
        elif attempt == 2:
            return ' '.join(parts[1:])  # Remove first word
        elif attempt == 3 and len(parts) > 2:
            return ' '.join(parts[:len(parts)//2] + parts[len(parts)//2+1:])  # Remove middle word
    return query

def find_best_match_address(query):
    original_query = query
    attempts = 0
    max_attempts = MAX_TRY * 3  # 3 different ways to broaden query

    while attempts < max_attempts:
        broaden_attempt = (attempts % 3) + 1
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

        query = broaden_query(query, broaden_attempt)
        attempts += 1

    print("No matches.")
    return None

# def addr2coord(address):
#     match_result = find_best_match_address(address)
#     if match_result:
#         return (round(float(match_result['LATITUDE']), 4), round(float(match_result['LONGITUDE']), 4))
#     else:
#         return (0, 0)


def addr2coord(address):
    match_result = find_best_match_address(address)
    if match_result:
        return {
            'address': match_result['ADDRESS'],
            'coords': (round(float(match_result['LATITUDE']), 4), round(float(match_result['LONGITUDE']), 4)),
            # 'confidence': match_result.get('confidence', 0)  # Add confidence if needed
        }
    else:
        return {
            'address': address,
            'coords': (0, 0),
            # 'confidence': 0
        }
    
def coord2addr(coord):
    lat, lon = coord
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('display_name', 'Address not found')
    return 'Address not found'

    

