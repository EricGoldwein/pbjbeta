import datetime
from typing import List, Dict, Optional

def generate_pbj_report(facility_name: str, location: str, ccn: str, affiliate_entity: str, 
                       review_period: str, dates_of_interest: List[datetime.datetime], 
                       metrics: Dict[str, float], quarterly_data: Optional[Dict] = None, 
                       state_comparison_data: Optional[Dict] = None, key_dates_data: Optional[Dict] = None) -> str:
    """
    Generate a PBJ report in HTML format based on the West Side template.
    """
    # Calculate ranges from quarterly data if available
    if quarterly_data:
        direct_care_values = [float(data['direct_care_hprd']) for data in quarterly_data.values()]
        rn_values = [float(data['direct_rn_hppd']) for data in quarterly_data.values()]
        min_hppd = min(direct_care_values) if direct_care_values else 0
        max_hppd = max(direct_care_values) if direct_care_values else 0
        min_rn_hppd = min(rn_values) if rn_values else 0
        max_rn_hppd = max(rn_values) if rn_values else 0
    else:
        # Fallback to metrics
        min_hppd = metrics.get('min_hppd', 0)
        max_hppd = metrics.get('max_hppd', 0)
        min_rn_hppd = metrics.get('min_rn_hppd', 0)
        max_rn_hppd = metrics.get('max_rn_hppd', 0)
    
    min_rating = metrics.get('min_rating', 1)
    max_rating = metrics.get('max_rating', 5)
    
    # Generate quarterly data table rows
    quarterly_rows = ""
    if quarterly_data is not None:
        for quarter, data in quarterly_data.items():
            quarterly_rows += f"""
                <tr style='height:11.45pt'>
                    <td width=60 nowrap valign=bottom style='width:45.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><span
                    style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:black'>{quarter}</span></p>
                    </td>
                    <td width=36 nowrap valign=bottom style='width:27.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('census', '-')}</span></p>
                    </td>
                    <td width=42 nowrap valign=bottom style='width:31.5pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('direct_care_hprd', '-')}</span></p>
                    </td>
                    <td width=60 nowrap valign=bottom style='width:45.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('reported_total_hprd', '-')}</span></p>
                    </td>
                    <td width=41 nowrap valign=bottom style='width:30.8pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('case_mix_total_hprd', '-')}</span></p>
                    </td>
                    <td width=51 nowrap valign=bottom style='width:38.35pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('direct_rn_hppd', '-')}</span></p>
                    </td>
                    <td width=51 nowrap valign=bottom style='width:38.3pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('reported_total_rn_hprd', '-')}</span></p>
                    </td>
                    <td width=51 nowrap colspan=2 valign=bottom style='width:38.35pt;background:
                    #D9D9D9;padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('case_mix_rn_hprd', '-')}</span></p>
                    </td>
                    <td width=51 nowrap valign=bottom style='width:38.35pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('staffing_rating', '-')}</span></p>
                    </td>
                    <td width=48 nowrap valign=bottom style='width:36.35pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('overall_rating', '-')}</span></p>
                    </td>
                    <td width=54 nowrap valign=bottom style='width:40.3pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('health_inspection_rating', '-')}</span></p>
                    </td>
                    <td width=54 nowrap valign=bottom style='width:40.7pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('total_staff_turnover', '-')}</span></p>
                    </td>
                    <td width=54 nowrap valign=bottom style='width:40.5pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.45pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('rn_turnover', '-')}</span></p>
                    </td>
                </tr>
            """
    
    # Generate state comparison table rows
    state_comparison_rows = ""
    if state_comparison_data is not None:
        for quarter, data in state_comparison_data.items():
            state_comparison_rows += f"""
                <tr style='height:11.0pt'>
                    <td width=64 nowrap valign=bottom style='width:48.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.0pt'>
                    <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><span
                    style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:black'>{quarter}</span></p>
                    </td>
                    <td width=64 nowrap valign=bottom style='width:48.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.0pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('facility_direct_care_hppd', '-')}</span></p>
                    </td>
                    <td width=64 nowrap valign=bottom style='width:48.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.0pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('state_direct_care_hppd', '-')}</span></p>
                    </td>
                    <td width=64 nowrap valign=bottom style='width:48.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.0pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('facility_direct_rn_hppd', '-')}</span></p>
                    </td>
                    <td width=56 nowrap valign=bottom style='width:42.0pt;background:#D9D9D9;
                    padding:0in 5.4pt 0in 5.4pt;height:11.0pt'>
                    <p class=MsoNormal align=right style='margin-bottom:0in;text-align:right;
                    line-height:normal'><span style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;
                    color:black'>{data.get('state_direct_rn_hppd', '-')}</span></p>
                    </td>
                </tr>
            """
    
    # Format dates of interest with detailed breakdowns
    dates_section = ""
    for i, date in enumerate(dates_of_interest):
        date_str = date.strftime('%Y-%m-%d')
        if key_dates_data and date_str in key_dates_data:
            data = key_dates_data[date_str]
            census = data['census']
            direct_care = data['direct_care_hppd']
            total_hppd = data['total_hppd']
            rn_hppd = data['rn_hppd']
            total_rn_hppd = data['total_rn_hppd']
        else:
            census = direct_care = total_hppd = rn_hppd = total_rn_hppd = 'N/A'
            
        # Create CMS PBJ data link for this specific date
        # Import the function from dynamic_facility_dashboard
        from dynamic_facility_dashboard import generate_pbj_source_link
        
        # Determine the quarter for this date
        year = date.year
        quarter_num = (date.month - 1) // 3 + 1
        quarter = f"{year}Q{quarter_num}"
        
        # Generate the specific CMS PBJ link for this date and facility
        cms_pbj_link = generate_pbj_source_link(quarter, date.strftime('%Y-%m-%d'), ccn, "nurse")
        if not cms_pbj_link:
            # Fallback to generic link if specific link generation fails
            cms_pbj_link = f"https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
        
        dates_section += f"""
            <p class="MsoListParagraphCxSpFirst" style="text-indent:-.25in">
                <span style="font-family:Symbol">•<span style="font:7.0pt 'Times New Roman'">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></span>
                <a href="{cms_pbj_link}"><b>{date.strftime('%A, %B %d, %Y')} (Key Event)</b></a>
            </p>
            <p class="MsoListParagraphCxSpMiddle" style="margin-left:1.0in;text-indent:-.25in">
                <span style="font-family:'Courier New'">o<span style="font:7.0pt 'Times New Roman'">&nbsp;&nbsp;</span></span>
                Census: {census}
            </p>
            <p class="MsoListParagraphCxSpMiddle" style="margin-left:1.0in;text-indent:-.25in">
                <span style="font-family:'Courier New'">o<span style="font:7.0pt 'Times New Roman'">&nbsp;&nbsp;</span></span>
                Direct Care: {direct_care} HPPD; Total: {total_hppd} HPPD
            </p>
            <p class="MsoListParagraphCxSpMiddle" style="margin-left:1.0in;text-indent:-.25in">
                <span style="font-family:'Courier New'">o<span style="font:7.0pt 'Times New Roman'">&nbsp;&nbsp;</span></span>
                RN: {rn_hppd} HPPD; Total RN: {total_rn_hppd} HPPD
            </p>
        """
    
    report_html = f"""<html>

<head>
<meta http-equiv=Content-Type content="text/html; charset=windows-1252">
<meta name=Generator content="Microsoft Word 15 (filtered)">
<style>
<!--
 /* Font Definitions */
 @font-face
	{{font-family:Wingdings;
	panose-1:5 0 0 0 0 0 0 0 0 0;}}
@font-face
	{{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;}}
@font-face
	{{font-family:Aptos;}}
@font-face
	{{font-family:"Aptos Narrow";}}
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:8.0pt;
	margin-left:0in;
	line-height:115%;
	font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
p.MsoCaption, li.MsoCaption, div.MsoCaption
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:10.0pt;
	margin-left:0in;
	font-size:9.0pt;
	font-family:"Aptos",sans-serif;
	color:#0E2841;
	font-style:italic;}}
a:link, span.MsoHyperlink
	{{color:#467886;
	text-decoration:underline;}}
p.MsoListParagraph, li.MsoListParagraph, div.MsoListParagraph
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:8.0pt;
	margin-left:.5in;
	line-height:115%;
	font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
p.MsoListParagraphCxSpFirst, li.MsoListParagraphCxSpFirst, div.MsoListParagraphCxSpFirst
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:0in;
	margin-left:.5in;
	line-height:115%;
	font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
p.MsoListParagraphCxSpMiddle, li.MsoListParagraphCxSpMiddle, div.MsoListParagraphCxSpMiddle
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:0in;
	margin-left:.5in;
	line-height:115%;
	font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
p.MsoListParagraphCxSpLast, li.MsoListParagraphCxSpLast, div.MsoListParagraphCxSpLast
	{{margin-top:0in;
	margin-right:0in;
	margin-bottom:8.0pt;
	margin-left:.5in;
	line-height:115%;
	font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
.MsoChpDefault
	{{font-size:12.0pt;
	font-family:"Aptos",sans-serif;}}
.MsoPapDefault
	{{margin-bottom:8.0pt;
	line-height:115%;}}
 /* Page Definitions */
 @page WordSection1
	{{size:8.5in 11.0in;
	margin:1.0in 1.0in 1.0in 1.0in;}}
div.WordSection1
	{{page:WordSection1;}}
 /* List Definitions */
 ol
	{{margin-bottom:0in;}}
ul
	{{margin-bottom:0in;}}
-->
</style>

</head>

<body lang=EN-US link="#467886" vlink="#96607D" style='word-wrap:break-word'>

<div class=WordSection1>

<p class=MsoNormal align=center style='text-align:center'><b><span
style='font-size:18.0pt;line-height:115%;color:#0070C0'>PBJ Brief: {facility_name}</span></b></p>

<p class=MsoNormal align=center style='text-align:center'><b>{location} ({ccn}) | Affiliate Entity: {affiliate_entity}<br>
</b>Review Period: {review_period}</p>

<p class=MsoNormal><i>The following brief includes a summary, quarterly data, and
dates of interest, with sources and methodology in the final section.</i></p>

<p class=MsoNormal><b><br>
</b><b><span style='font-size:14.0pt;line-height:115%;color:#156082'>PBJ Summary,
{facility_name}</span></b><span style='font-size:14.0pt;line-height:115%;
color:#156082'><br>
</span>{facility_name} reported quarterly direct care staffing ratios ranging
from <b>{min_hppd:.2f} to {max_hppd:.2f} Hours Per Patient Day (HPPD) </b>and RN staffing ratios
ranging from <b>{min_rn_hppd:.2f} to {max_rn_hppd:.2f} HPPD </b>between {review_period}. Though
{facility_name} Total staffing ratios were consistently above case-mix (expected)
staffing levels and above state levels, the facility's RN
staffing (excl. Admin, DON) were lower than state levels. During this period,
the facility's staffing ratings ranged from {min_rating} to {max_rating} stars.<br>
<span style='color:#156082'><br>
</span><b><span style='font-size:14.0pt;line-height:115%;color:#156082'>Quarterly
Data: {review_period}</span></b></p>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 width=654
 style='border-collapse:collapse'>
 <tr style='height:35.9pt'>
  <td width=60 valign=bottom style='width:45.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Matching
  Quarter*</span></b></p>
  </td>
  <td width=36 valign=bottom style='width:27.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Avg.
  Census</span></b></p>
  </td>
  <td width=42 valign=bottom style='width:31.5pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Direct
  Care HPRD</span></b></p>
  </td>
  <td width=60 valign=bottom style='width:45.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Reported
  Total HPRD</span></b></p>
  </td>
  <td width=41 valign=bottom style='width:30.8pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Case-Mix
  Total HPRD</span></b></p>
  </td>
  <td width=51 valign=bottom style='width:38.35pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Direct
  RN HPPD</span></b></p>
  </td>
  <td width=58 colspan=2 valign=bottom style='width:43.35pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Reported
  Total RN HPRD</span></b></p>
  </td>
  <td width=44 valign=bottom style='width:33.3pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Case-Mix
  RN HPRD</span></b></p>
  </td>
  <td width=51 valign=bottom style='width:38.35pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Staffing
  Rating</span></b></p>
  </td>
  <td width=48 valign=bottom style='width:36.35pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Overall
  Rating</span></b></p>
  </td>
  <td width=54 valign=bottom style='width:40.3pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Health
  Inspection Rating</span></b></p>
  </td>
  <td width=54 valign=bottom style='width:40.7pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>Total
  Staff Turnover</span></b></p>
  </td>
  <td width=54 valign=bottom style='width:40.5pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#E97132;padding:0in 5.4pt 0in 5.4pt;height:35.9pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>RN
  Turnover</span></b></p>
  </td>
 </tr>
 {quarterly_rows}
</table>

<p class=MsoCaption style='page-break-after:avoid'><span style='font-size:8.0pt;
color:windowtext'>Note: *Provider Info staffing data matched to Payroll-Based
Journal data; Overall and Health Inspection ratings not necessarily updated
quarterly and may not match exact quarter; Direct Care staffing metrics via PBJ
data, all other data via Provider Info.</span></p>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 align=left
 width=312 style='width:3.25in;border-collapse:collapse;margin-left:6.75pt;
 margin-right:6.75pt'>
 <tr style='height:26.0pt'>
  <td width=64 valign=bottom style='width:48.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#156082;padding:0in 5.4pt 0in 5.4pt;height:26.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>CY_Qtr</span></b></p>
  </td>
  <td width=64 valign=bottom style='width:48.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#156082;padding:0in 5.4pt 0in 5.4pt;height:26.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>{facility_name} Direct Care HPPD</span></b></p>
  </td>
  <td width=64 valign=bottom style='width:48.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#156082;padding:0in 5.4pt 0in 5.4pt;height:26.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>State Direct Care HPPD</span></b></p>
  </td>
  <td width=64 valign=bottom style='width:48.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#156082;padding:0in 5.4pt 0in 5.4pt;height:26.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>{facility_name} Direct RN HPPD</span></b></p>
  </td>
  <td width=56 valign=bottom style='width:42.0pt;border-top:solid black 1.0pt;
  border-left:none;border-bottom:solid black 1.0pt;border-right:none;
  background:#156082;padding:0in 5.4pt 0in 5.4pt;height:26.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;line-height:normal'><b><span
  style='font-size:8.0pt;font-family:"Aptos Narrow",sans-serif;color:white'>State Direct RN HPPD</span></b></p>
  </td>
 </tr>
 {state_comparison_rows}
</table>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<b><span style='font-size:14.0pt;line-height:115%;font-family:"Aptos",sans-serif;
color:#156082'><br clear=all style='page-break-before:always'>
</span></b>

<p class=MsoNormal><b><span style='font-size:14.0pt;line-height:115%;
color:#156082'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;line-height:115%;
color:#156082'>Dates of interest: </span></b></p>

{dates_section}

<p class=MsoNormal><b><br>
</b><b><span style='font-size:14.0pt;line-height:115%;color:#156082'>Sources
and Methodology</span></b></p>

<p class=MsoListParagraphCxSpFirst style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><b><span
style='font-size:11.0pt;line-height:115%'>Sources</span></b><span
style='font-size:11.0pt;line-height:115%'>: </span><a
href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"><span
style='font-size:11.0pt;line-height:115%'>CMS Payroll-Based Journal</span></a><span
style='font-size:11.0pt;line-height:115%'>; </span><a
href="https://data.cms.gov/provider-data/dataset/4pq5-n9py"><span
style='font-size:11.0pt;line-height:115%'>Provider Data Catalogue</span></a><span
style='font-size:11.0pt;line-height:115%'>; </span><a
href="https://data.cms.gov/quality-of-care/nursing-home-chain-performance-measures"><span
style='font-size:11.0pt;line-height:115%'>Nursing Home Chain Performance
Measures</span></a><span style='font-size:11.0pt;line-height:115%'>; </span><a
href="http://pbjdashboard.com/?facility={ccn}"><span style='font-size:11.0pt;line-height:115%'>PBJ
Dashboard (320 Consulting)</span></a></p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><b><span
style='font-size:11.0pt;line-height:115%'>Direct Care</span></b><span
style='font-size:11.0pt;line-height:115%'> refers to RN, LPN, CNA, MedAide Tech,
Nurse Aide in Training; it excludes RN Admin, RN DON, LPN Admin.</span></p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><b><span
style='font-size:11.0pt;line-height:115%'>Total Nurse Staff</span></b><span
style='font-size:11.0pt;line-height:115%'> refers to all nurse staff positions:
RN, RN Admin, RN DON, LPN, LPN Admin, Med Aide/Tech, Nurse Aide in Training.</span></p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><b><span
style='font-size:11.0pt;line-height:115%'>Total RN </span></b><span
style='font-size:11.0pt;line-height:115%'>includes RN, RN Admin, and RN DON. </span></p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><b><span
style='font-size:11.0pt;line-height:115%'>State staffing levels </span></b><span
style='font-size:11.0pt;line-height:115%'>determined based on aggregate
staffing hours and census.</span></p>

<p class=MsoListParagraphCxSpLast style='text-indent:-.25in'><span
style='font-size:11.0pt;line-height:115%;font-family:Symbol'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span><a
href="https://www.macpac.gov/publication/state-policies-related-to-nursing-facility-staffing/"><span
style='font-size:11.0pt;line-height:115%'>According to MACPAC (2022)</span></a><span
style='font-size:11.0pt;line-height:115%'>, Alabama state staffing levels are determined based on aggregate staffing hours and census data from CMS Payroll-Based Journal.</span></p>

</div>

</body>

</html>"""
    
    return report_html

# Example usage
if __name__ == "__main__":
    facility_name = "West Side House"
    location = "Worcester, MA"
    ccn = "225500"
    affiliate_entity = "Elder Services"
    review_period = "April 2021 - May 2023"
    dates_of_interest = [datetime.datetime(2021, 4, 1), datetime.datetime(2021, 7, 1), 
                        datetime.datetime(2022, 1, 1), datetime.datetime(2022, 7, 1), 
                        datetime.datetime(2023, 1, 1)]
    metrics = {
        "min_hppd": 3.44,
        "max_hppd": 4.17,
        "min_rn_hppd": 0.148,
        "max_rn_hppd": 0.422,
        "min_rating": 1,
        "max_rating": 5
    }
    report = generate_pbj_report(facility_name, location, ccn, affiliate_entity, review_period, dates_of_interest, metrics)
    print(report)
