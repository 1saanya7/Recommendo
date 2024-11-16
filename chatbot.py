import requests


def get_recommendations(user_input, category, language, additional_info=None):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyA6A6JObugkjN-Vbxtqu0QS4W4Sc3OpsSY"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    category_prompts = {
        "movies": f"Recommend some movies based on: {user_input}",
        "books": f"Suggest books based on: {user_input}",
        "songs": f"Give me song recommendations for: {user_input} in {language}. {additional_info if additional_info else ''}"
    }


    prompt = category_prompts.get(category, f"Provide recommendations for: {user_input}")
    

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
      
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  
        return response.json() 
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling the API: {e}")
        return None


def extract_recommendations(response):
    if response and 'candidates' in response:
        candidates = response['candidates']
        recommendations = []
        for candidate in candidates:
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            for part in parts:
                text = part.get('text', "")
         
                recommendations.append(text.strip())
        return recommendations
    return []


if __name__ == "__main__":

    category = input("Enter category (movies/books/songs): ").strip().lower()
    user_input = input(f"Enter your preferences for {category}: ")
    language = input("Enter preferred language for recommendations: ").strip().lower()

    additional_info = None
    if category == 'songs':
        additional_info = input("Do you have any specific mood or theme (e.g., heartbreak, melancholy, etc.)? ").strip()
    
 
    result = get_recommendations(user_input, category, language, additional_info)
    
    if result:
        
        recommendations = extract_recommendations(result)
        if recommendations:
            print("\nHere are your recommendations:")
            for i, recommendation in enumerate(recommendations, 1):
                print(f"{i}. {recommendation}")
        else:
            print("No recommendations found.")
