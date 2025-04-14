from fpdf import FPDF
from datetime import date


def create_presentation_pdf(blog_title, blog_content):
    """Create a PDF presentation from structured slide content"""
    # Create landscape PDF
    pdf = FPDF(orientation='L', format='A4')
    pdf.set_auto_page_break(auto=False)

    # Normalize text to be Latin-1 compatible
    blog_title = normalize_text(blog_title)
    blog_content = normalize_text(blog_content)

    # Parse the slide content
    sections = parse_slide_content(blog_content)

    # Create title slide
    create_title_slide(pdf, blog_title)

    # Create content slides (limit to 7 slides)
    for i, section in enumerate(sections[:7]):
        create_content_slide(pdf, section, i + 1)

    # Create ending slide
    create_ending_slide(pdf, blog_title)

    # Get PDF as bytes
    output = pdf.output()

    # Ensure we return bytes, not bytearray
    if isinstance(output, bytearray):
        return bytes(output)
    return output


def normalize_text(text):
    """Replace non-Latin-1 characters with their Latin-1 equivalents"""
    # Replace smart quotes with straight quotes
    text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    # Replace bullet points with hyphens
    text = text.replace('•', '-').replace('●', '-').replace('·', '-')
    # Replace em/en dashes with hyphens
    text = text.replace('—', '-').replace('–', '-')
    # Replace ellipsis with three dots
    text = text.replace('…', '...')
    return text


def parse_slide_content(content):
    """Parse pre-formatted slide content with defined sections and bullets"""
    sections = []

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
                    bullet_points.append(normalize_text(point.replace("*", "")))

        # Create section if we have title and bullets
        if title and bullet_points:
            sections.append({
                "title": normalize_text(title),
                "bullet_points": bullet_points,
            })

    return sections


def create_title_slide(pdf, title):
    """Create the title slide"""
    pdf.add_page()

    # Set background color (light blue-gray)
    pdf.set_fill_color(230, 230, 242)
    pdf.rect(0, 0, pdf.w, pdf.h, style='F')

    # Title
    pdf.set_text_color(51, 51, 128)  # Dark blue
    pdf.set_font('Arial', 'B', 36)

    # Center the title
    pdf.set_y(pdf.h / 2 - 30)
    pdf.cell(0, 20, title, 0, 1, 'C')

    # Subtitle
    pdf.set_font('Arial', '', 24)
    pdf.cell(0, 20, "Blog Summary Presentation", 0, 1, 'C')

    # Footer
    pdf.set_font('Arial', '', 12)
    pdf.set_y(pdf.h - 20)
    pdf.cell(90, 10, "AI Blog Content Creator", 0, 0, 'L')

    # Date
    today = date.today().strftime("%B %d, %Y")
    pdf.cell(0, 10, today, 0, 0, 'R')


def create_content_slide(pdf, section, slide_number):
    """Create a content slide with title and bullets"""
    pdf.add_page()

    # Set background color (white)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, pdf.w, pdf.h, style='F')

    # Header background (blue)
    pdf.set_fill_color(51, 77, 179)
    pdf.rect(0, 0, pdf.w, 40, style='F')

    # Title
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font('Arial', 'B', 24)

    title = section["title"]
    if len(title) > 50:
        title = title[:47] + "..."

    pdf.set_xy(20, 15)
    pdf.cell(0, 10, title, 0, 1)

    # Bullet points
    pdf.set_text_color(0, 0, 0)  # Black text
    pdf.set_font('Arial', '', 18)
    pdf.set_y(60)

    for bullet in section["bullet_points"]:
        pdf.set_x(30)

        # Format bullet point with a latin-1 compatible bullet
        bullet_text = f"- {bullet}"  # Using hyphen instead of bullet point

        # Use multi_cell for automatic line wrapping
        pdf.multi_cell(0, 12, bullet_text)
        pdf.ln(8)  # Add space between bullet points

        # Break if we're running out of space
        if pdf.get_y() > pdf.h - 50:
            break

    # Footer with slide number
    pdf.set_fill_color(51, 77, 179)  # Blue
    pdf.rect(0, pdf.h - 20, pdf.w, 20, style='F')

    pdf.set_text_color(255, 255, 255)  # White
    pdf.set_font('Arial', '', 12)
    pdf.set_xy(20, pdf.h - 15)
    pdf.cell(90, 10, "AI Blog Content Creator", 0, 0)
    pdf.cell(0, 10, f"Slide {slide_number}", 0, 0, 'R')


def create_ending_slide(pdf, title):
    """Create a summary/ending slide"""
    pdf.add_page()

    # Set background color (light blue-gray)
    pdf.set_fill_color(230, 230, 242)  # Light blue-gray
    pdf.rect(0, 0, pdf.w, pdf.h, style='F')

    # Title
    pdf.set_text_color(51, 51, 128)  # Dark blue
    pdf.set_font('Arial', 'B', 30)

    # Thank you text
    pdf.set_y(pdf.h / 2 - 30)
    pdf.cell(0, 20, "Thank You!", 0, 1, 'C')

    # Blog title reminder
    pdf.set_font('Arial', '', 20)
    reminder = f"Summary of: {title}"
    if len(reminder) > 60:
        reminder = reminder[:57] + "..."
    pdf.cell(0, 20, reminder, 0, 1, 'C')

    # Footer
    pdf.set_font('Arial', '', 12)
    pdf.set_y(pdf.h - 20)
    pdf.cell(90, 10, "AI Blog Content Creator", 0, 0, 'L')

    # Date
    today = date.today().strftime("%B %d, %Y")
    pdf.cell(0, 10, today, 0, 0, 'R')