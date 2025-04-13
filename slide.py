# slide.py
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
import re
import io


def create_presentation_pdf(blog_title, blog_content):
    """Create a PDF presentation from structured slide content"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)

    # Parse the structured content to extract slides
    sections = parse_slide_content(blog_content)

    # Create title slide
    create_title_slide(c, blog_title, width, height)
    c.showPage()

    # Create content slides (limit to 7 slides)
    for i, section in enumerate(sections[:7]):
        create_content_slide(c, section, width, height, i + 1)
        c.showPage()

    # Create ending slide
    create_ending_slide(c, blog_title, width, height)
    c.showPage()

    c.save()
    return buffer.getvalue()


def parse_slide_content(content):
    """Parse pre-formatted slide content with defined sections and bullets"""
    sections = []
    current_section = None

    # Split the content by slide markers
    slide_blocks = content.split("**Slide ")

    for block in slide_blocks:
        if not block.strip():
            continue

        # Extract slide title and content
        parts = block.split("*", 1)
        if len(parts) < 2:
            continue

        # The title is in format: "N: Title**"
        title_part = parts[0].strip()
        if ":" in title_part:
            title = title_part.split(":", 1)[1].strip()
        else:
            title = title_part

        # Extract bullet points - they start with *
        bullet_points = []
        for line in parts[1].strip().split('\n'):
            line = line.strip()
            if line.startswith('*'):
                # Clean the bullet point
                point = line[1:].strip()
                if point:
                    bullet_points.append(point.replace("*", ""))

        # Create section if we have title and bullets
        if title and bullet_points:
            sections.append({
                "title": title,
                "bullet_points": bullet_points,
                "content": []  # Keep this for compatibility
            })

    return sections


def create_content_slide(c, section, width, height, slide_number):
    """Create a content slide with title, bullets, and image description"""
    # Background
    c.setFillColorRGB(1, 1, 1)  # White
    c.rect(0, 0, width, height, fill=True)

    # Header background
    c.setFillColorRGB(0.2, 0.3, 0.7)  # Blue
    c.rect(0, height - 80, width, 80, fill=True)

    # Title
    c.setFillColorRGB(1, 1, 1)  # White text on blue background
    c.setFont("Helvetica-Bold", 24)
    title = section["title"]
    if len(title) > 50:
        title = title[:47] + "..."
    c.drawString(40, height - 50, title)

    # Bullet points
    y_position = height - 120
    c.setFillColorRGB(0, 0, 0)  # Black
    c.setFont("Helvetica", 18)

    for bullet in section["bullet_points"]:
        # Handle long bullet points with line wrapping
        bullet_text = f"• {bullet}"

        # Line wrapping for long bullets
        max_width = width - 100
        words = bullet_text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            if c.stringWidth(test_line, "Helvetica", 18) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long by itself, just add it
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(' '.join(current_line))

        # Draw the wrapped bullet point lines
        for line in lines:
            if y_position < 100:  # Avoid drawing below a certain point
                break
            c.drawString(60, y_position, line)
            y_position -= 30

        y_position -= 20  # Add spacing between bullet points

    # Footer with slide number
    c.setFillColorRGB(0.2, 0.3, 0.7)  # Blue
    c.rect(0, 0, width, 30, fill=True)
    c.setFillColorRGB(1, 1, 1)  # White
    c.setFont("Helvetica", 12)
    c.drawString(30, 10, "AI Blog Content Creator")
    c.drawString(width - 70, 10, f"Slide {slide_number}")

def create_title_slide(c, title, width, height):
    """Create the title slide"""
    # Background
    c.setFillColorRGB(0.9, 0.9, 0.95)  # Light blue-gray
    c.rect(0, 0, width, height, fill=True)

    # Title
    c.setFillColorRGB(0.2, 0.2, 0.5)  # Dark blue
    c.setFont("Helvetica-Bold", 36)

    # Center the title text
    title_width = c.stringWidth(title, "Helvetica-Bold", 36)
    title_x = (width - title_width) / 2
    c.drawString(title_x, height/2 + 30, title)

    # Subtitle
    c.setFont("Helvetica", 24)  # Using standard Helvetica
    subtitle = "Blog Summary Presentation"
    subtitle_width = c.stringWidth(subtitle, "Helvetica", 24)
    c.drawString((width - subtitle_width) / 2, height/2 - 30, subtitle)

    # Footer
    c.setFont("Helvetica", 12)
    c.drawString(30, 30, "AI Blog Content Creator")

    # Date
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    date_width = c.stringWidth(today, "Helvetica", 12)
    c.drawString(width - date_width - 30, 30, today)


def create_ending_slide(c, title, width, height):
    """Create a summary/ending slide"""
    # Background
    c.setFillColorRGB(0.9, 0.9, 0.95)  # Light blue-gray
    c.rect(0, 0, width, height, fill=True)

    # Title
    c.setFillColorRGB(0.2, 0.2, 0.5)  # Dark blue
    c.setFont("Helvetica-Bold", 30)
    end_title = "Thank You!"
    title_width = c.stringWidth(end_title, "Helvetica-Bold", 30)
    c.drawString((width - title_width) / 2, height/2 + 40, end_title)

    # Blog title reminder
    c.setFont("Helvetica", 20)
    reminder = f"Summary of: {title}"
    if len(reminder) > 60:
        reminder = reminder[:57] + "..."
    reminder_width = c.stringWidth(reminder, "Helvetica", 20)
    c.drawString((width - reminder_width) / 2, height/2 - 20, reminder)

    # Footer
    c.setFont("Helvetica", 12)
    c.drawString(30, 30, "AI Blog Content Creator")

    # Date
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    date_width = c.stringWidth(today, "Helvetica", 12)
    c.drawString(width - date_width - 30, 30, today)