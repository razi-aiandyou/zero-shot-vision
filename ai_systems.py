from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_mercedes_car(base64_image):
    car_identification_steps = """
    1. Look for the Logo:
       - Check the front grille, hood, trunk, and wheels for the iconic Mercedes-Benz star logo, which is a three-pointed star enclosed in a circle.
    2. Examine the Front Grille:
       - Mercedes cars often have a distinctive front grille design. Many models feature horizontal slats with the logo prominently displayed in the center.
    3. Check the Headlights:
       - Mercedes-Benz cars usually have advanced and uniquely shaped headlights. Look for sleek, angular designs with integrated LED strips.
    4. Inspect the Rear Design :
       - Look at the tail lights and rear bumper. Mercedes cars often have stylish, elongated tail lights with a modern, sleek appearance.
    5. Side Profile and Lines:
       - Mercedes cars tend to have elegant and smooth side profiles with flowing lines. Look for a sophisticated and aerodynamic shape.
    6. Model Badges:
       - Look for model names and badges. Mercedes-Benz uses specific naming conventions (e.g., C-Class, E-Class, S-Class) that are often found on the rear of the vehicle.
    7. Wheel Design:
       - Mercedes-Benz wheels often have a distinct design with the Mercedes logo in the center cap. They may feature unique patterns and high-quality finishes.
    8. Overall Build Quality:
       - Mercedes-Benz cars are known for their high build quality and luxurious appearance. Look for premium materials, precise panel gaps, and a solid, high-end look.
    """

    specific_mercedes_car_list = """
    Here are the Mercedes-Benz car models and their specific and unique features available in Indonesia:

    ### Electric Models
    1. **EQA, EQB, EQC, EQE, EQS, EQV**
       - **Mercedes-Benz EQA**: A compact SUV with a sleek, modern design. It features a closed-off front grille with a star pattern, LED light strips, and blue accents to signify its electric nature.
       - **Mercedes-Benz EQB**: This electric SUV has a boxier design with a closed-off grille, LED light strips, and a robust, practical look. It often includes roof rails and a spacious interior.
       - **Mercedes-Benz EQC**: A mid-sized SUV with a bold front grille, integrated LED light bars, and aerodynamic design elements. The EQC emphasizes a luxury feel with a minimalist interior.
       - **Mercedes-Benz EQE**: A sedan with a streamlined design, flush door handles, and a seamless, flowing front grille. The EQE features advanced LED lighting technology and a futuristic interior.
       - **Mercedes-Benz EQS**: The flagship electric sedan with a luxurious and aerodynamic design. It features a panoramic roof, large wheels, and advanced digital lighting. The EQS offers a spacious and ultra-modern interior.
       - **Mercedes-Benz EQV**: An electric MPV with a versatile and spacious design, closed-off front grille, and distinctive EQV badging. It is designed for both luxury and practicality.

    ### AMG Models
    2. **A 35 AMG, CLA 45 AMG, GLA 45 AMG, C 43 AMG, E 53 AMG, G 63 AMG**
       - **Mercedes-AMG A 35**: A sporty compact hatchback with aggressive front bumper, large air intakes, and quad exhaust tips. It features AMG-specific design elements and badging.
       - **Mercedes-AMG CLA 45**: A compact, performance-oriented sedan with a distinctive AMG grille, aerodynamic enhancements, and quad exhaust pipes. Look for sporty details and AMG badges.
       - **Mercedes-AMG GLA 45**: A compact SUV with a bold design, large wheels, and AMG-specific styling elements like a more aggressive front bumper and quad exhaust pipes.
       - **Mercedes-AMG C 43**: A mid-sized sedan with a sporty, yet refined design. It features AMG-specific details like a unique front grille, larger wheels, and quad exhaust tips.
       - **Mercedes-AMG E 53**: A luxury sedan with a performance twist, featuring AMG-specific design elements such as a more aggressive front bumper, larger wheels, and distinctive exhaust tips.
       - **Mercedes-AMG G 63**: A high-performance SUV with a rugged, boxy design, AMG-specific grille, large wheels, and side-exit exhaust pipes. It combines luxury and off-road capability.

    ### Standard Models
    3. **A 200, B 200, C 200, E 200, S 450**
       - **Mercedes-Benz A 200**: A compact hatchback or sedan with a sleek design, diamond-patterned grille, and sporty elements. It offers a modern and youthful appearance.
       - **Mercedes-Benz B 200**: A compact MPV with a practical and spacious design. It features a bold front grille, large windows, and a comfortable interior layout.
       - **Mercedes-Benz C 200**: A mid-sized luxury sedan with a refined and elegant design. It features a distinctive front grille, LED headlights, and a sleek profile.
       - **Mercedes-Benz E 200**: A larger luxury sedan with an elegant and sophisticated design. Look for a prominent front grille, advanced LED lighting, and a spacious interior.
       - **Mercedes-Benz S 450**: A full-sized luxury sedan with a stately presence. It features a large grille, advanced lighting technology, and an opulent interior.

    ### SUVs
    4. **GLA 200, GLB 200, GLC 200, GLE 450, GLS 450**
       - **Mercedes-Benz GLA 200**: A compact SUV with a sporty design, distinctive front grille, and rugged elements. It offers a balance of luxury and practicality.
       - **Mercedes-Benz GLB 200**: A compact SUV with a boxier design, providing ample space and a robust look. It features practical elements like roof rails and a spacious interior.
       - **Mercedes-Benz GLC 200**: A mid-sized SUV with a refined design, featuring a prominent front grille, sleek profile, and luxurious interior details.
       - **Mercedes-Benz GLE 450**: A larger SUV with a bold and sophisticated design. It features advanced lighting technology, a spacious interior, and a balance of luxury and performance.
       - **Mercedes-Benz GLS 450**: A full-sized luxury SUV with a commanding presence. It features a large grille, advanced lighting, and a spacious, luxurious interior.

    By following these steps and paying attention to the specific design elements and badges, you can accurately identify the make and model of the Mercedes-Benz cars listed in the image.
    """

    prompt = f"""
    Your task is to classify if the car in the user picture is a Mercedes Car Brand or not by using the steps for identifying and classifying Mercedes cars and Mercedes car features provided below as a reference to help your classification. Here are your guidelines:
    1. Provide a concise response indicating your level of accuracy of detecting the image. Include the car's make, model, year, and color in two sentences. Format the response as: 'The car is a Mercedes car with [accuracy of detecting]% accuracy of detecting image. The model is a [year] [color] [make and model].
    2. If your certainty level for the classification result is below 90%, kindly tell the user that they need to provide a better image with a better angle.
    3. Try to focus your classification task on the largest car in the user image in terms of size and aspect ratio of the image.
    4. If multiple cars are detected in the user picture that hinders your classification process and result, do not classify the car make and model. Instead, kindly tell the user that their image has multiple cars and they need to retake the picture to include only a single car.
    5. If the angle of the user picture hinders your classification task, kindly instruct the user to retake the picture from a better angle.
    6. If the user picture quality is hindering your classification task, kindly instruct the user to retake the picture again.
    7. If the car in the user image is not a Mercedes car, simply state, "The car in your picture is not a Mercedes.", additionally, include your best guess of the car's make, model, year, and color in the response.
    
    Shape your response as if you're talking with the user, e.g., "I can see a classic car in your picture, but I'm not sure if it's a Mercedes or not. Can you retake the car picture from a better angle?".
   
    Steps for identifying and classifying Mercedes cars:
    {car_identification_steps}

    Mercedes car features:
    {specific_mercedes_car_list}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a car identifier and classifier for Mercedes Models with the persona of being Formal and Friendly."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content