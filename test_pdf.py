from fix_pdf import get_db_connection, get_facility_info, get_quarterly_metrics
from fpdf import FPDF
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.colors import grey, whitesmoke, beige, black
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_historical_quarters(current_quarter: str, num_quarters: int = 4) -> list:
    """Get a list of historical quarters including and before the current quarter."""
    match = re.match(r'(\d{4})Q(\d)', current_quarter)
    if not match:
        return []
    
    year = int(match.group(1))
    quarter = int(match.group(2))
    
    quarters = []
    for _ in range(num_quarters):
        quarters.append(f"{year}Q{quarter}")
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
    
    return quarters

def get_historical_metrics(provnum: str, conn) -> pd.DataFrame:
    """Get historical metrics for a facility."""
    try:
        # Read from the CSV file instead of querying DuckDB
        csv_file = f'facility_{provnum}_test.csv'
        if not os.path.exists(csv_file):
            print(f"No data file found for facility {provnum}")
            return pd.DataFrame()
            
        df = pd.read_csv(csv_file)
        df = df[['CY_QTR', 'Total_HPRD', 'RN_HPRD', 'Nurse_Assistant_HPRD', 'Total_Resident_Days']]
        df = df.rename(columns={
            'Total_HPRD': 'TOTAL_HRD',
            'RN_HPRD': 'RN_HRD',
            'Nurse_Assistant_HPRD': 'NURSE_ASSISTANT_HRD',
            'Total_Resident_Days': 'TOTAL_RESIDENT_DAYS'
        })
        return df.sort_values('CY_QTR', ascending=False).head(8)
    except Exception as e:
        print(f"Error getting historical metrics: {e}")
        return pd.DataFrame()

def get_facility_citations(provnum: str, conn) -> pd.DataFrame:
    """Get recent citations for a facility."""
    # Since we don't have a citations table yet, return empty DataFrame
    # This can be implemented once we have the citations data
    print(f"Citations data not yet available for facility {provnum}")
    return pd.DataFrame(columns=['survey_date', 'deficiency_category', 'severity_level', 'description'])

def generate_trend_summary(historical_metrics: pd.DataFrame) -> str:
    """Generate a summary of historical trends."""
    if historical_metrics.empty:
        return "No historical data available."
    
    summary = []
    metrics = {
        'Total_HPRD': 'Total staffing',
        'RN_HPRD': 'RN staffing',
        'Nurse_Assistant_HPRD': 'Nurse assistant'
    }
    
    # Sort by date ascending to ensure correct trend calculation
    historical_metrics = historical_metrics.sort_values('CY_QTR', ascending=True)
    
    for metric_col, metric_name in metrics.items():
        if metric_col in historical_metrics.columns:
            values = historical_metrics[metric_col].dropna()
            if len(values) > 1:
                oldest_value = values.iloc[0]  # First value (oldest)
                newest_value = values.iloc[-1]  # Last value (newest)
                change = ((newest_value - oldest_value) / oldest_value * 100)
                
                # Format the trend description
                trend = "increased" if change > 0 else "decreased"
                summary.append(f"{metric_name} has {trend} by {abs(change):.1f}% (from {oldest_value:.2f} to {newest_value:.2f} hours per resident day)")
    
    return "\n".join(summary) if summary else "No significant trends detected."

def format_historical_for_prompt(df: pd.DataFrame) -> str:
    """Format historical data for the AI prompt."""
    if df.empty:
        return "No historical data available."
    
    display_df = df[['CY_QTR', 'Total_HPRD', 'RN_HPRD', 'Nurse_Assistant_HPRD']].copy()
    display_df.columns = ['Quarter', 'Total HPRD', 'RN HPRD', 'Nurse Assistant HPRD']
    return display_df.to_csv(index=False)

def validate_ai_response(response: str, context: dict) -> bool:
    """Validate that the AI's response matches the data exactly."""
    try:
        # Extract metrics from the response
        metrics = {
            'total_hprd': context['current_metrics']['total_hprd'],
            'rn_hprd': context['current_metrics']['rn_hprd'],
            'nurse_assistant_hprd': context['current_metrics']['nurse_assistant_hprd']
        }
        
        # Check each metric is mentioned correctly
        for metric_name, value in metrics.items():
            # Convert to string with 2 decimal places for comparison
            value_str = f"{value:.2f}"
            if value_str not in response:
                print(f"Warning: {metric_name} value {value_str} not found in response")
                return False
        
        # Verify quarter references
        if context['quarter'] not in response:
            print(f"Warning: Current quarter {context['quarter']} not referenced in response")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating AI response: {e}")
        return False

def generate_ai_insights(facility_info, historical_metrics, state_quarter, citations=None):
    """Generate AI insights about facility performance using OpenAI."""
    try:
        # Get the most recent quarter's data (first row since we sorted descending)
        current_quarter_data = historical_metrics.iloc[0]  # Most recent quarter
        quarter_match = re.match(r'(\d{4})Q(\d)', current_quarter_data['CY_QTR'])
        if quarter_match:
            year, quarter = quarter_match.groups()
            quarter_display = f"Q{quarter} {year}"
        else:
            quarter_display = current_quarter_data['CY_QTR']

        # Get the time period for trends (last row since we sorted descending)
        oldest_quarter = historical_metrics.iloc[-1]['CY_QTR']
        oldest_match = re.match(r'(\d{4})Q(\d)', oldest_quarter)
        if oldest_match:
            oldest_year, oldest_quarter = oldest_match.groups()
            oldest_display = f"Q{oldest_quarter} {oldest_year}"
        else:
            oldest_display = oldest_quarter

        # Prepare the context for the AI
        context = {
            'facility_name': facility_info['name'],
            'state': facility_info['state'],
            'quarter': quarter_display,
            'oldest_quarter': oldest_display,
            'current_metrics': {
                'total_hprd': current_quarter_data['Total_HPRD'],
                'rn_hprd': current_quarter_data['RN_HPRD'],
                'nurse_assistant_hprd': current_quarter_data['Nurse_Assistant_HPRD']
            },
            'state_averages': {
                'total_hprd': state_quarter['Total_HPRD'].iloc[0],
                'rn_hprd': state_quarter['RN_HPRD'].iloc[0],
                'nurse_assistant_hprd': state_quarter['Nurse_Assistant_HPRD'].iloc[0]
            },
            'historical_trend': historical_metrics[['CY_QTR', 'Total_HPRD', 'RN_HPRD', 'Nurse_Assistant_HPRD']].to_dict('records'),
            'has_citations': citations is not None and not citations.empty
        }

        # Create prompt for GPT
        prompt = f"""Analyze the following nursing home facility data for {quarter_display} and provide insights:

Facility: {context['facility_name']} in {context['state']}

Current Metrics for {quarter_display} vs State Averages:
- Total Hours per Resident Day: {context['current_metrics']['total_hprd']:.2f} (State: {context['state_averages']['total_hprd']:.2f})
- RN Hours per Resident Day: {context['current_metrics']['rn_hprd']:.2f} (State: {context['state_averages']['rn_hprd']:.2f})
- Nurse Care Hours per Resident Day: {context['current_metrics']['nurse_assistant_hprd']:.2f} (State: {context['state_averages']['nurse_assistant_hprd']:.2f})

Trend Summary (from {oldest_display} to {quarter_display}):
{generate_trend_summary(historical_metrics)}

Historical Data (all available quarters):
{format_historical_for_prompt(historical_metrics)}

{'Facility has recent severe citations (G or above).' if context['has_citations'] else 'No severe citations found.'}

Please provide a comprehensive analysis in the following format:

Current Performance Analysis ({quarter_display})
[Provide a clear analysis of current performance compared to state averages. Always reference the specific quarter being analyzed.]

taffing Trends ({oldest_display} to {quarter_display})
[Analyze staffing level trends over time, starting with the most recent quarter {quarter_display} and comparing to {oldest_display}]

Areas for Improvement ({quarter_display})
[List 2-3 specific areas needing attention, based on {quarter_display} data]

Recommendations
[Provide 2-3 actionable recommendations based on the current {quarter_display} performance]

Keep the analysis professional and factual. Do not use markdown formatting or special characters. Format section titles in bold and all caps as shown above. Always ensure you are referencing the correct quarter's data in your analysis. Do not assume facility with high staffing ratio is providing appropriate staffing; a ratio above the state average does not imply sufficient staffing"""

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"You are a healthcare analytics expert specializing in nursing home performance analysis. You are analyzing data for {quarter_display}. Always reference this specific quarter in your analysis and ensure all metrics cited match the provided data exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        ai_response = response.choices[0].message.content
        
        # Validate the response
        if not validate_ai_response(ai_response, context):
            # If validation fails, try once more with a stronger emphasis on accuracy
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": f"You are a healthcare analytics expert specializing in nursing home performance analysis. You MUST use EXACT numbers from the provided data, rounded to 2 decimal places. You are analyzing data for {quarter_display}."},
                    {"role": "user", "content": "Your previous response did not match the data exactly. Please ensure all metrics are quoted precisely.\n\n" + prompt}
                ],
                temperature=0.5  # Lower temperature for more precise output
            )
            ai_response = response.choices[0].message.content

        return ai_response

    except Exception as e:
        print(f"Error generating AI insights: {str(e)}")
        return f"Unable to generate AI insights for {quarter_display} at this time. Please review the data manually."

class EnhancedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(25, 20, 25)  # Reduced top margin from 30 to 20
        self.set_auto_page_break(auto=True, margin=25)
        self._current_section = None
        
        # Default to Arial/Helvetica since Times New Roman might not be available
        self.set_font('Arial', '')
        
    def header(self):
        # Add more space at top of first page
        top_margin = 20 if self.page_no() == 1 else 15  # Reduced from 25/20 to 20/15
        self.set_y(top_margin)
        
        # Add colored title bar with precise positioning
        self.set_fill_color(51, 122, 183)  # Professional blue
        self.rect(0, top_margin, 210, 12, 'F')  # Reduced height from 20 to 12
        
        # Add report title with exact centering
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        title = 'Nursing Home Report'
        title_w = self.get_string_width(title)
        self.set_x((210 - title_w) / 2)
        self.cell(title_w, 12, title, 0, 1, 'C', False)  # Reduced height from 20 to 12
        
        # Reset text color and add generation date with time and attribution
        self.set_text_color(0, 0, 0)
        if self.page_no() == 1:
            self.set_font('Arial', '', 10)
            self.set_text_color(128, 128, 128)
            current_time = datetime.now()
            date_str = f"Generated on {current_time.strftime('%B %d, %Y at %I:%M %p')} by 320 Consulting"
            self.cell(0, 6, date_str, 0, 1, 'R')
            self.set_text_color(0, 0, 0)
        
    def footer(self):
        self.set_y(-20)
        
        # Add separator line
        self.set_draw_color(200, 200, 200)
        self.line(25, self.get_y(), 185, self.get_y())
        self.ln(1)
        
        # Footer text with sources
        self.set_font('Arial', '', 9)
        self.set_text_color(128, 128, 128)
        
        # Sources and page number
        self.cell(140, 5, 'Sources: CMS Payroll Based Journal Daily Nurse Staffing; CMS Health Deficiencies (03/2025)', 0, 0, 'L')
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', 0, 0, 'R')

    def add_section_header(self, title, is_subsection=False):
        """Add a section header with consistent styling."""
        # Check if we need a page break
        if self.get_y() > 250:
            self.add_page()
        
        # Add some space before section
        if self.get_y() > 40:
            self.ln(4)
        
        # Style based on section type
        if is_subsection:
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(240, 240, 240)  # Lighter gray for subsections
            self.cell(0, 8, title, 0, 1, 'L', True)
            self.ln(2)  # Less space after subsections
        else:
            self.set_font('Arial', 'B', 14)
            self.set_fill_color(245, 245, 245)
            self.cell(0, 9, title.title(), 0, 1, 'L', True)
            self.ln(3)
        
        self._current_section = title
        
    def add_metric_table(self, headers, data, col_widths, alternating_colors=True, center_align=False):
        """Add a table with improved formatting."""
        # Table headers with dark gray background
        self.set_fill_color(240, 240, 240)
        self.set_font('Arial', 'B', 11)
        
        # Calculate x position based on alignment preference
        total_width = sum(col_widths)
        if center_align:
            start_x = (210 - 2*self.l_margin - total_width) / 2 + self.l_margin
        else:
            start_x = self.l_margin
        self.set_x(start_x)
        
        # Draw header cells
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        self.ln()
        
        # Table data with alternating colors
        self.set_font('Arial', '', 11)
        for row_idx, row in enumerate(data):
            self.set_x(start_x)  # Reset x position for each row
            if alternating_colors and row_idx % 2 == 1:
                self.set_fill_color(248, 248, 248)
                fill = True
            else:
                self.set_fill_color(255, 255, 255)
                fill = False
            
            for i, value in enumerate(row):
                align = 'R' if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').replace('+', '').isdigit()) else 'L'
                self.cell(col_widths[i], 6, str(value), 1, 0, align, fill)
            self.ln()
        
    def add_paragraph(self, text, is_justified=True, bullet=False):
        """Add a paragraph with improved bullet points and spacing."""
        self.set_font('Arial', '', 11)
        
        # Calculate bullet indent and position
        left_margin = self.l_margin
        if bullet:
            self.set_left_margin(left_margin + 8)
        
        # Handle text with numbers but avoid creating new bullets
        text = text.strip()
        
        # Regular bullet point or paragraph
        lines = text.split('\n')
        for line_idx, line in enumerate(lines):
            if bullet and line_idx == 0:
                bullet_x = self.get_x() - 6
                bullet_y = self.get_y()
                self.draw_bullet(bullet_x, bullet_y)
                self.set_x(self.get_x() + 2)
            
            # Ensure text is properly encoded
            line = line.strip().encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, line, 0, 'J' if is_justified else 'L')
            bullet = False  # Only first line gets bullet
        
        # Reset margin and add minimal spacing
        self.set_left_margin(left_margin)
        self.ln(2 if bullet else 3)

    def draw_bullet(self, x, y):
        """Draw a small filled square as a bullet point."""
        self.set_fill_color(0, 0, 0)  # Black color for bullet
        self.rect(x + 2, y + 2, 1.5, 1.5, 'F')  # Small filled square
        self.set_fill_color(255, 255, 255)  # Reset fill color to white

    def add_section_divider(self):
        """Add a subtle divider line between sections."""
        self.ln(3)  # Reduced spacing
        self.set_draw_color(220, 220, 220)  # Lighter gray
        self.line(25, self.get_y(), 185, self.get_y())
        self.ln(3)  # Reduced spacing

def title_case(text):
    """Custom title case that keeps 'and', 'of', 'at', 'for' lowercase."""
    lowercase_words = {'and', 'of', 'at', 'the', 'in', 'on', 'with', 'to', 'for', 'by', 'as', 'but', 'or', 'nor', 'yet', 'so'}
    words = text.split()
    titled_words = []
    
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in lowercase_words:
            titled_words.append(word.title())
        else:
            titled_words.append(word.lower())
    
    return ' '.join(titled_words)

def get_days_in_quarter(year: int, quarter: int) -> int:
    """Calculate the exact number of days in a quarter, accounting for leap years."""
    if quarter == 1:  # Jan-Mar
        # Check if it's a leap year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return 31 + (29 if is_leap else 28) + 31  # Jan + Feb + Mar
    elif quarter == 2:  # Apr-Jun
        return 30 + 31 + 30  # Apr + May + Jun
    elif quarter == 3:  # Jul-Sep
        return 31 + 31 + 30  # Jul + Aug + Sep
    else:  # Oct-Dec
        return 31 + 30 + 31  # Oct + Nov + Dec

def create_enhanced_pdf_report(provnum, selected_quarter, conn=None):
    """Create an enhanced PDF report with professional formatting."""
    try:
        # Parse the selected quarter
        year = int(selected_quarter[:4])
        quarter = int(selected_quarter[-1])
        
        # Format display strings
        quarter_display = f"Q{quarter} {year}"
        oldest_display = f"Q{quarter-3} {year-1}" if quarter > 3 else f"Q{quarter+1} {year-1}"
        
        # Read facility data from CSV
        facility_df = pd.read_csv(f'facility_{provnum}_test.csv')
        if facility_df.empty:
            raise ValueError(f"No data found for facility {provnum}")
            
        # Read ownership entity data
        entity_df = pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv')
        
        # Get facility info from most recent record
        latest_record = facility_df.sort_values('CY_QTR', ascending=False).iloc[0]
        facility_info = {
            'name': title_case(latest_record['PROVNAME']),
            'state': latest_record['STATE'],
            'provnum': provnum
        }
        
        # Get ownership entity data
        entity_data = None
        try:
            # First try to find the entity by provider number
            if 'PROVNUM' in entity_df.columns:
                entity_matches = entity_df[entity_df['PROVNUM'] == provnum]
                if not entity_matches.empty:
                    entity_data = entity_matches.iloc[0]
            
            # If no direct match by provider number, try to find by facility name
            if entity_data is None:
                # Clean the facility name for better matching
                clean_facility_name = facility_info['name'].lower().strip()
                entity_matches = entity_df[entity_df['Affiliated entity'].str.lower().str.contains(clean_facility_name, na=False)]
                if not entity_matches.empty:
                    entity_data = entity_matches.iloc[0]
            
            # If still no match, try to find by partial name match
            if entity_data is None:
                # Split facility name into words and try to match any of them
                facility_words = clean_facility_name.split()
                for word in facility_words:
                    if len(word) > 3:  # Only try words longer than 3 characters
                        entity_matches = entity_df[entity_df['Affiliated entity'].str.lower().str.contains(word, na=False)]
                        if not entity_matches.empty:
                            entity_data = entity_matches.iloc[0]
                            break
        except Exception as e:
            print(f"Warning: Error looking up entity data: {str(e)}")
            entity_data = None
        
        # Get metrics for selected quarter
        quarter_data = facility_df[facility_df['CY_QTR'] == selected_quarter]
        if quarter_data.empty:
            raise ValueError(f"No metrics found for {provnum} in quarter {selected_quarter}")
            
        # Get state averages
        state_df = pd.read_csv('state_quarterly_metrics.csv')
        # Standardize state dataframe column names
        state_mapping = {
            'CY_QTR': 'CY_QTR',
            'cy_qtr': 'CY_QTR',
            'CY_Qtr': 'CY_QTR',
            'STATE': 'STATE',
            'state': 'STATE',
            'Total_HPRD': 'Total_HPRD',
            'total_hprd': 'Total_HPRD',
            'Total_HRD': 'Total_HPRD',
            'total_hrd': 'Total_HPRD',
            'RN_HPRD': 'RN_HPRD',
            'rn_hprd': 'RN_HPRD',
            'RN_HRD': 'RN_HPRD',
            'rn_hrd': 'RN_HPRD',
            'Nurse_Care_HPRD': 'Nurse_Care_HPRD',
            'nurse_care_hprd': 'Nurse_Care_HPRD',
            'Nurse_Care_HRD': 'Nurse_Care_HPRD',
            'nurse_care_hrd': 'Nurse_Care_HPRD',
            'Nurse_Assistant_HPRD': 'Nurse_Assistant_HPRD',
            'nurse_assistant_hprd': 'Nurse_Assistant_HPRD',
            'Nurse_Assistant_HRD': 'Nurse_Assistant_HPRD',
            'nurse_assistant_hrd': 'Nurse_Assistant_HPRD'
        }
        state_df = state_df.rename(columns=state_mapping)
        
        state_quarter = state_df[
            (state_df['STATE'] == facility_info['state']) & 
            (state_df['CY_QTR'] == selected_quarter)
        ]
        
        # Get all historical metrics (sorted newest to oldest)
        historical_metrics = facility_df.dropna(subset=['CY_QTR', 'Total_HPRD', 'RN_HPRD', 'Nurse_Assistant_HPRD'])
        historical_metrics = historical_metrics.sort_values('CY_QTR', ascending=False)
        
        # Get citations data
        citations_df = pd.read_csv('NH_HealthCitations_Mar2025.csv', low_memory=False)
        # Standardize citations column names
        citations_mapping = {
            'CMS Certification Number (CCN)': 'PROVNUM',
            'CMS Certification Number': 'PROVNUM',
            'CCN': 'PROVNUM',
            'PROVNUM': 'PROVNUM',
            'provnum': 'PROVNUM',
            'Survey Date': 'SURVEY_DATE',
            'Deficiency Tag Number': 'TAG',
            'Scope Severity Code': 'SEVERITY',
            'Deficiency Description': 'DEFICIENCY_DESC'
        }
        citations_df = citations_df.rename(columns=citations_mapping)
        
        # Get all facility citations, sorted by date
        facility_citations = citations_df[citations_df['PROVNUM'] == provnum].sort_values('SURVEY_DATE', ascending=False)
        
        # Generate AI insights once and store them
        print("Generating AI insights...")
        ai_insights = generate_ai_insights(facility_info, historical_metrics, state_quarter, facility_citations)
        print("AI Insights generated:", ai_insights)  # Debug print
        
        # Initialize PDF with custom class
        pdf = EnhancedPDF()
        pdf.alias_nb_pages()  # Enable page numbering
        pdf.add_page()
        
        # Facility Information Section - Simplified format
        facility_name = title_case(facility_info['name'])
        pdf.set_font('Arial', 'B', 14)
        # Set text color to dark blue (RGB: 0, 0, 139)
        pdf.set_text_color(0, 0, 139)
        # Center the header
        title = f"Report for {facility_name} ({provnum}), {facility_info['state']}"
        title_width = pdf.get_string_width(title)
        pdf.set_x((210 - title_width) / 2)  # Center on page (A4 width is 210mm)
        pdf.cell(title_width, 10, title, 0, 1, 'C')
        # Reset text color to black for rest of document
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        # Current Quarter Performance Section - Updated title
        pdf.add_section_header(f"Q{selected_quarter[5]} {selected_quarter[:4]} Performance")
        metrics_headers = ["Metric", "Facility", f"{facility_info['state']}", "Variance"]
        metrics_data = [
            ["Total HPRD", f"{quarter_data['Total_HPRD'].iloc[0]:.2f}", 
             f"{state_quarter['Total_HPRD'].iloc[0]:.2f}", 
             f"{(quarter_data['Total_HPRD'].iloc[0] - state_quarter['Total_HPRD'].iloc[0]):+.2f}"],
            ["RN HPRD", f"{quarter_data['RN_HPRD'].iloc[0]:.2f}", 
             f"{state_quarter['RN_HPRD'].iloc[0]:.2f}", 
             f"{(quarter_data['RN_HPRD'].iloc[0] - state_quarter['RN_HPRD'].iloc[0]):+.2f}"],
            ["Nurse Assistant HPRD", f"{quarter_data['Nurse_Assistant_HPRD'].iloc[0]:.2f}", 
             f"{state_quarter['Nurse_Assistant_HPRD'].iloc[0]:.2f}", 
             f"{(quarter_data['Nurse_Assistant_HPRD'].iloc[0] - state_quarter['Nurse_Assistant_HPRD'].iloc[0]):+.2f}"]
        ]
        pdf.add_metric_table(metrics_headers, metrics_data, [50, 35, 35, 35])
        pdf.ln(5)
        
        # Citations Section with improved formatting
        if not facility_citations.empty:
            pdf.add_section_header("Facility Citations")
            
            for idx, citation in facility_citations.iterrows():
                # Add divider line between citations
                if idx > 0:
                    pdf.ln(2)
                    pdf.set_draw_color(220, 220, 220)  # Light gray
                    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
                    pdf.ln(2)
                
                # Create a visual block with light gray background
                pdf.set_fill_color(248, 248, 248)  # Very light gray
                pdf.rect(25, pdf.get_y(), 160, 20, 'F')
                
                # Determine severity color
                is_severe = citation['SEVERITY'][0] in 'GHIJKL'
                severity_color = (255, 0, 0) if is_severe else (0, 0, 0)
                
                # Format citation header with fixed-width spacing
                pdf.set_font('Courier', 'B', 10)  # Use fixed-width font
                
                # Survey date, tag, and severity on one line with fixed spacing
                date_str = f"Survey Date: {citation['SURVEY_DATE']}"
                tag_str = f"Tag: F{citation['TAG']}"  # Added 'F' prefix to tag number
                severity_str = f"Severity: {citation['SEVERITY']}"
                
                # Print with fixed spacing
                pdf.cell(60, 6, date_str, 0, 0)
                pdf.cell(5, 6, "|", 0, 0, 'C')
                pdf.cell(40, 6, tag_str, 0, 0)
                pdf.cell(5, 6, "|", 0, 0, 'C')
                
                # Severity with color
                pdf.set_text_color(*severity_color)
                pdf.cell(0, 6, severity_str, 0, 1)
                pdf.set_text_color(0, 0, 0)
                
                # Citation description with proper encoding
                pdf.set_font('Arial', '', 10)
                pdf.set_y(pdf.get_y() + 2)  # Add small padding
                pdf.multi_cell(0, 5, citation['DEFICIENCY_DESC'].encode('latin-1', 'replace').decode('latin-1'), 0, 'J')
                
                # Add padding after citation
                pdf.ln(3)
        
        # Add Ownership Entity Section after citations
        if entity_data is not None:
            # Add page break before Ownership Entity section
            pdf.add_page()
            
            # Update title format with proper entity ID formatting
            entity_id = str(int(float(entity_data['Affiliated entity ID']))) if entity_data['Affiliated entity ID'] else "N/A"
            entity_title = f"Ownership Entity: {title_case(entity_data['Affiliated entity'])} ({entity_id})"
            
            # Create a table for entity data with left-justified values
            entity_headers = ["Metric", "Value"]
            entity_data_rows = [
                ["Number of Facilities", f"{entity_data['Number of facilities']:,}"],
                ["States/Territories", f"{entity_data['Number of states and territories with operations']}"],
                ["Special Focus Facilities", f"{entity_data['Number of Special Focus Facilities (SFF)']}"],
                ["SFF Candidates", f"{entity_data['Number of SFF candidates']}"],
                ["Facilities with Abuse Icon", f"{entity_data['Number of facilities with an abuse icon']} ({entity_data['Percentage of facilities with an abuse icon']}%)"],
                ["For-Profit Facilities", f"{entity_data['Percent of facilities classified as for-profit']}%"],
                ["Overall 5-Star Rating", f"{entity_data['Average overall 5-star rating']}"],
                ["Health Inspection Rating", f"{entity_data['Average health inspection rating']}"],
                ["Staffing Rating", f"{entity_data['Average staffing rating']}"],
                ["Quality Rating", f"{entity_data['Average quality rating']}"],
                ["Total Nurse Hours/Resident Day", f"{entity_data['Average total nurse hours per resident day']}"],
                ["Weekend Nurse Hours/Resident Day", f"{entity_data['Average total weekend nurse hours per resident day']}"],
                ["RN Hours/Resident Day", f"{entity_data['Average total Registered Nurse hours per resident day']}"],
                ["Nursing Staff Turnover", f"{entity_data['Average total nursing staff turnover percentage']}%"],
                ["RN Turnover", f"{entity_data['Average Registered Nurse turnover percentage']}%"],
                ["Administrators Left", f"{entity_data['Average number of administrators who have left the nursing home']}"],
                ["Total Fines", f"${entity_data['Total amount of fines in dollars']:,.2f}"],
                ["Average Fines per Facility", f"${entity_data['Average amount of fines in dollars']:,.2f}"]
            ]
            
            # Add the entity section
            pdf.add_section_header(entity_title)
            
            # Use larger column widths and left alignment for both columns
            pdf.add_metric_table(entity_headers, entity_data_rows, [80, 80], center_align=False)
            pdf.ln(10)

            # Generate and add ownership context analysis
            print("Generating ownership context analysis...")
            ownership_analysis = generate_ownership_context_analysis(
                facility_info,
                entity_data,
                quarter_data.iloc[0],
                state_quarter.iloc[0]
            )
            print("Ownership context analysis generated")

            # Add ownership context analysis section
            pdf.add_section_header("Ownership Context Analysis")
            pdf.set_font('Arial', '', 10)
            pdf.set_left_margin(25)
            pdf.set_right_margin(25)
            pdf.multi_cell(0, 6, ownership_analysis)
            pdf.ln(10)
        else:
            # Add a note if no entity data was found
            pdf.add_section_header("Ownership Entity Information")
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "No ownership entity data available for this facility.", ln=True)
            pdf.ln(5)
        
        # Add Analysis section on a new page
        pdf.add_page()
        pdf.add_section_header("Analysis")
        pdf.set_font('Arial', '', 10)
        
        if not ai_insights:
            print("Warning: No AI insights generated")
            pdf.multi_cell(0, 6, "Unable to generate analysis at this time. Please try again later.")
        else:
            # Split the AI insights into sections
            sections = ai_insights.split('\n\n')
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Find the first newline to separate title from content
                newline_pos = section.find('\n')
                if newline_pos > 0:
                    title = section[:newline_pos].strip()
                    content = section[newline_pos + 1:].strip()
                    
                    # Handle Unicode characters in title
                    title = title.encode('latin-1', 'replace').decode('latin-1')
                    
                    # Add title in bold
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 6, title, ln=True)
                    
                    # Handle Unicode characters in content
                    content = content.encode('latin-1', 'replace').decode('latin-1')
                    
                    # Add content with proper margins and word wrap
                    pdf.set_font('Arial', '', 10)
                    pdf.set_left_margin(25)  # Set left margin
                    pdf.set_right_margin(25)  # Set right margin
                    pdf.set_x(25)  # Ensure we start at the left margin
                    pdf.multi_cell(160, 6, content)  # Use 160mm width (210mm - 25mm margins on each side)
                    pdf.ln(3)
                else:
                    # If no newline found, handle Unicode characters and add the whole section
                    section = section.encode('latin-1', 'replace').decode('latin-1')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_left_margin(25)  # Set left margin
                    pdf.set_right_margin(25)  # Set right margin
                    pdf.set_x(25)  # Ensure we start at the left margin
                    pdf.multi_cell(160, 6, section)  # Use 160mm width
                    pdf.ln(3)
        
        pdf.ln(10)
        
        # Staffing Trends Section (on new page)
        pdf.add_page()
        pdf.add_section_header("Staffing Trends")
        trend_headers = ["Quarter", "Total", "RN", "Nurse Care", "Nurse Asst", "Contract %", "Avg Census"]
        trend_data = []
        
        # Sort historical metrics in descending order for the table
        for _, row in historical_metrics.iterrows():
            # Convert quarter format from 2017Q1 to Q1 2017
            quarter_match = re.match(r'(\d{4})Q(\d)', row['CY_QTR'])
            if quarter_match:
                year, quarter = map(int, quarter_match.groups())
                formatted_quarter = f"Q{quarter} {year}"
                days_in_quarter = get_days_in_quarter(year, quarter)
            else:
                formatted_quarter = row['CY_QTR']
                days_in_quarter = 90  # Fallback if quarter format is unexpected
            
            # Calculate average MDS Census (Total Resident Days / Days in Quarter)
            avg_census = row['Total_Resident_Days'] / days_in_quarter
                
            trend_data.append([
                formatted_quarter,
                f"{row['Total_HPRD']:.2f}",
                f"{row['RN_HPRD']:.2f}",
                f"{row['Nurse_Care_HPRD']:.2f}",
                f"{row['Nurse_Assistant_HPRD']:.2f}",
                f"{row['Contract_Staff_Percentage']:.1f}%",
                f"{avg_census:.1f}"
            ])
        
        # Adjust column widths for better spacing
        col_widths = [30, 25, 25, 25, 25, 25, 35]  # Adjusted widths for all columns
        pdf.add_metric_table(trend_headers, trend_data, col_widths, alternating_colors=True, center_align=True)
        
        # Add Ownership Context Analysis as the final section on a new page
        if entity_data is not None:
            pdf.add_page()
            pdf.add_section_header("Ownership Context Analysis")
            
            # Generate ownership context analysis
            print("Generating ownership context analysis...")
            ownership_analysis = generate_ownership_context_analysis(
                facility_info,
                entity_data,
                quarter_data.iloc[0],
                state_quarter.iloc[0]
            )
            print("Ownership context analysis generated")
            
            # Add the analysis with proper formatting
            pdf.set_font('Arial', '', 10)
            pdf.set_left_margin(25)
            pdf.set_right_margin(25)
            pdf.multi_cell(0, 6, ownership_analysis)
            pdf.ln(10)
        
        # Save the PDF
        output_file = f"enhanced_report_{provnum}_{selected_quarter}.pdf"
        pdf.output(output_file)
        return output_file
        
    except Exception as e:
        print(f"Error creating enhanced PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_pdf_generation():
    """Test the enhanced PDF report generation."""
    try:
        # Verify required files exist
        required_files = {
            'facility_data': 'facility_335386_test.csv',
            'state_data': 'state_quarterly_metrics.csv',
            'citations_data': 'NH_HealthCitations_Mar2025.csv'
        }
        
        for file_type, filepath in required_files.items():
            if not os.path.exists(filepath):
                print(f"Missing required file: {filepath}")
                return
        
        provnum = "335386"  # Test facility
        quarter = "2024Q3"  # Test quarter
        
        # Create test connection (not actually used but required by function signature)
        conn = None
        
        output_file = create_enhanced_pdf_report(provnum, quarter, conn)
        if output_file:
            print(f"Enhanced PDF report generated successfully: {output_file}")
            
            # Print report contents summary
            df = pd.read_csv(required_files['facility_data'])
            facility_name = df.iloc[0]['PROVNAME']
            print(f"\nReport generated for:")
            print(f"Facility: {facility_name} ({provnum})")
            print(f"Quarter: {quarter}")
            print("\nReport sections:")
            print("1. Facility Information")
            print("2. Current Quarter Metrics (with state comparisons)")
            print("3. Staffing Trends")
            if os.path.exists(required_files['citations_data']):
                print("4. Facility Citations")
            print("5. Analysis and Recommendations")
        else:
            print("Failed to generate enhanced PDF report")
            
    except Exception as e:
        print(f"Error in test_enhanced_pdf_generation: {str(e)}")
        import traceback
        traceback.print_exc()

def clean_special_characters(text):
    """Replace special characters with their ASCII equivalents."""
    replacements = {
        '\u2014': '-',  # em dash
        '\u2013': '-',  # en dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201C': '"',  # left double quote
        '\u201D': '"',  # right double quote
        '\u2026': '...',  # ellipsis
        '\u00A0': ' ',  # non-breaking space
    }
    for special, ascii_char in replacements.items():
        text = text.replace(special, ascii_char)
    return text

def generate_ownership_context_analysis(facility_info, entity_data, facility_metrics, state_metrics):
    """Generate AI analysis placing facility in context of ownership entity data."""
    try:
        # Prepare the context for the AI with safe field access
        context = {
            'facility_name': facility_info['name'],
            'entity_name': title_case(entity_data.get('Affiliated entity', 'Unknown Entity')),
            'facility_total_hprd': facility_metrics.get('Total_HPRD', 0),
            'entity_avg_hprd': entity_data.get('Average total nurse hours per resident day', 0),
            'facility_rn_hprd': facility_metrics.get('RN_HPRD', 0),
            'entity_avg_rn_hprd': entity_data.get('Average total Registered Nurse hours per resident day', 0),
            'entity_sff_count': entity_data.get('Number of Special Focus Facilities (SFF)', 0),
            'entity_sff_candidates': entity_data.get('Number of SFF candidates', 0),
            'entity_abuse_icon_count': entity_data.get('Number of facilities with an abuse icon', 0),
            'entity_abuse_icon_percent': entity_data.get('Percentage of facilities with an abuse icon', 0),
            'entity_avg_staffing_rating': entity_data.get('Average staffing rating', 0),
            'entity_avg_overall_rating': entity_data.get('Average overall 5-star rating', 0)
        }

        # Create prompt for GPT
        prompt = f"""Analyze how {context['facility_name']} fits within the broader context of its ownership entity {context['entity_name']}. Consider the following data:

Facility Metrics vs Entity Averages:
- Total HPRD: {context['facility_total_hprd']:.2f} (Entity Avg: {context['entity_avg_hprd']:.2f})
- RN HPRD: {context['facility_rn_hprd']:.2f} (Entity Avg: {context['entity_avg_rn_hprd']:.2f})

Entity-wide Metrics:
- SFF Facilities: {context['entity_sff_count']}
- SFF Candidates: {context['entity_sff_candidates']}
- Facilities with Abuse Icon: {context['entity_abuse_icon_count']} ({context['entity_abuse_icon_percent']}%)
- Average Staffing Rating: {context['entity_avg_staffing_rating']}
- Average Overall Rating: {context['entity_avg_overall_rating']}

Provide a 250-word analysis that:
1. Discusses entity's summary data (total facilities, SFF, abuse, staffing rating, overall rating)
2. Compares facility's total and RN staffing levels to the entity's average and discusses implications

Keep the analysis professional and factual. Focus on data-driven insights and avoid speculation and judgement. Higher staffing or ratings does not indicate better. Use only ASCII characters (no special characters like em dashes or smart quotes)."""

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a healthcare analytics expert specializing in nursing home performance analysis. Provide a concise, data-driven analysis of how a facility fits within its ownership entity's portfolio. Use only ASCII characters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Clean any remaining special characters
        return clean_special_characters(response.choices[0].message.content)

    except Exception as e:
        print(f"Error generating ownership context analysis: {str(e)}")
        return "Unable to generate ownership context analysis at this time."

if __name__ == "__main__":
    test_enhanced_pdf_generation() 