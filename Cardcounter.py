import tkinter as tk
from tkinter import messagebox
import random
import cv2  # OpenCV for camera and image processing
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import torch  # PyTorch for YOLO inference

# Load the trained model
model = load_model('card_classifier.h5')

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace 'best.pt' with your trained model

# Define class labels (e.g., '2_of_hearts', 'king_of_spades', etc.)
class_labels = ['2_of_hearts', '3_of_hearts', ..., 'king_of_spades']  # Add all 52 card labels

def card_count(card_number):
    if 2 <= card_number <= 6:
        return 1
    elif 7 <= card_number <= 9:
        return 0
    elif card_number == 1 or 10 <= card_number <= 13:
        return -1
    else:
        raise ValueError("Invalid card number. Please enter a number between 1 and 13.")

def card_layout():
    return (
        "Card Layout:\n"
        "  +1: 2, 3, 4, 5, 6\n"
        "   0: 7, 8, 9\n"
        "  -1: 1 (Ace), 10, 11 (Jack), 12 (Queen), 13 (King)"
    )

def betting_suggestion(total_count):
    """Provide betting suggestions based on the count."""
    if total_count >= 5:
        return "Suggestion: Bet Big!"
    elif total_count <= -5:
        return "Suggestion: Minimum Bet."
    else:
        return "Suggestion: Neutral. Play cautiously."

def update_count():
    try:
        card_number = int(entry_card_number.get())
        if not (1 <= card_number <= 13):
            raise ValueError("Invalid card number. Please enter a number between 1 and 13.")
        count = card_count(card_number)
        update_count.total_count += count
        label_count.config(text=f"Count: {update_count.total_count}")
        label_suggestion.config(text=betting_suggestion(update_count.total_count))
        entry_card_number.delete(0, tk.END)
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

def simulate_cards():
    try:
        num_cards = int(entry_simulation.get())
        if num_cards <= 0:
            raise ValueError("Number of cards must be greater than 0.")
        
        simulation_count = 0
        cards_drawn = []
        for _ in range(num_cards):
            card_number = random.randint(1, 13)
            cards_drawn.append(card_number)
            count = card_count(card_number)
            simulation_count += count
        
        cards_display = ", ".join(map(str, cards_drawn))
        label_simulation_result.config(
            text=f"Simulation Result: Count after {num_cards} cards is {simulation_count}\n"
                 f"Cards Drawn: {cards_display}"
        )
        entry_simulation.delete(0, tk.END)
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

def classify_card(frame):
    """Classify the card in the given frame using YOLO."""
    # Perform inference
    results = yolo_model(frame)

    # Extract the detected card label
    if results.xyxy[0].size(0) > 0:  # Check if any object is detected
        card_label = results.pandas().xyxy[0]['name'][0]  # Get the first detected label
        return card_label
    else:
        return "No card detected"

def scan_cards():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Classify the card in the frame
        card_label = classify_card(frame)

        # Display the frame with the card label
        cv2.putText(frame, f"Card: {card_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Card Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def quit_program():
    root.destroy()

# Initialize total count as a function attribute
update_count.total_count = 0

# Create the main window
root = tk.Tk()
root.title("Card Counting Program")

# Layout
label_title = tk.Label(root, text="Card Counting Program", font=("Arial", 16))
label_title.pack(pady=10)

label_layout = tk.Label(root, text=card_layout(), font=("Arial", 12), justify="left")
label_layout.pack(pady=10)

label_count = tk.Label(root, text="Count: 0", font=("Arial", 14))
label_count.pack(pady=10)

label_suggestion = tk.Label(root, text="Suggestion: Neutral. Play cautiously.", font=("Arial", 12))
label_suggestion.pack(pady=10)

frame_input = tk.Frame(root)
frame_input.pack(pady=10)

label_input = tk.Label(frame_input, text="Enter card number (1-13):", font=("Arial", 12))
label_input.pack(side=tk.LEFT, padx=5)

entry_card_number = tk.Entry(frame_input, font=("Arial", 12), width=5)
entry_card_number.pack(side=tk.LEFT, padx=5)

button_submit = tk.Button(frame_input, text="Submit", font=("Arial", 12), command=update_count)
button_submit.pack(side=tk.LEFT, padx=5)

frame_simulation = tk.Frame(root)
frame_simulation.pack(pady=10)

label_simulation = tk.Label(frame_simulation, text="Run Simulation (Enter number of cards):", font=("Arial", 12))
label_simulation.pack(side=tk.LEFT, padx=5)

entry_simulation = tk.Entry(frame_simulation, font=("Arial", 12), width=5)
entry_simulation.pack(side=tk.LEFT, padx=5)

button_simulate = tk.Button(frame_simulation, text="Simulate", font=("Arial", 12), command=simulate_cards)
button_simulate.pack(side=tk.LEFT, padx=5)

label_simulation_result = tk.Label(root, text="", font=("Arial", 12))
label_simulation_result.pack(pady=10)

button_scan = tk.Button(root, text="Scan Cards", font=("Arial", 12), command=scan_cards)
button_scan.pack(pady=10)

button_quit = tk.Button(root, text="Quit", font=("Arial", 12), command=quit_program)
button_quit.pack(pady=10)

# Run the application
root.mainloop()