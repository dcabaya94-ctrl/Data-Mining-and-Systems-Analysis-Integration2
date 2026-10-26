import json, csv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
R=json.load(open('results.json')); m=R['model']
ss=getSampleStyleSheet()
B=ParagraphStyle('b',parent=ss['BodyText'],fontSize=9.5,leading=13)
S=ParagraphStyle('s',parent=B,fontSize=8,leading=10)
H1=ParagraphStyle('h1',parent=ss['Heading1'],fontSize=14,textColor=colors.HexColor('#2a78d6'),spaceBefore=8)
H2=ParagraphStyle('h2',parent=ss['Heading2'],fontSize=11,spaceBefore=6)
P=lambda t,s=B:Paragraph(t,s)
def tbl(rows,w,fs=8):
    t=Table([[Paragraph(str(c),ParagraphStyle('c',parent=S,fontSize=fs,leading=fs+2)) for c in r] for r in rows],colWidths=w,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eefb')),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]));return t
def img(f,w=6.3):
    from PIL import Image as I
    a,b=I.open(f).size; return Image(f,width=w*inch,height=w*inch*b/a)
dq=[['Issue','Rows','Action']]+[[a,b,c] for a,b,c in R['dq']]
uc=R['uc']; lt=R['lt']; fr=list(csv.reader(open('functional_requirements.csv')))
imp=R['importance']
E_=[]
E_+=[P('SAD Laboratory Examination: Data Mining and Systems Analysis Integration',ParagraphStyle('t',parent=ss['Title'],fontSize=17)),
P('Course: Systems Analysis and Design | BS Information Systems | Student: ____________ Section: ______ Date: ______'),
P('<b>Data note:</b> facility_reservations.csv was not supplied with the exam paper. All results come from a seeded <b>synthetic stand-in</b> dataset (make_dataset.py; 436 raw rows, no real student data) with injected defects. The notebook is fully re-runnable, so rerunning it on the official file updates every figure; numbers below must then be re-read from the notebook.',S),
P('Part I: Dataset acquisition and understanding',H1),
P('<b>Task 1.</b> Loaded with pandas: 436 rows x 13 columns (5 categorical, 5 numerical/ID-like, 2 date, 1 target source). Preview, dtypes and summary statistics are in the notebook (Part I). <b>Task 2.</b> Target: <i>Reservation_Outcome</i>. Predictors: Facility_Type, User_Category (categorical); Lead_Time_Days (derived), Participants, Duration_Hours (numerical); also Time_Slot, Day_of_Week, Equipment_Requested. Post-outcome fields (Usage_Status, Actual_Attendance, Cancellation_Days_Before) are not predictors. Full data dictionary: data_dictionary.csv.'),
P('<b>Task 3.</b> Data would be obtained with a read-only SQL query on the reservation table (e.g. SELECT ... FROM reservations JOIN facilities JOIN users WHERE reservation_date BETWEEN ...) or a paginated REST endpoint (GET /api/reservations?from=&amp;to=), exported to CSV with user identifiers anonymised. <b>Validation check:</b> reject or flag rows where Booking_Date is after Reservation_Date, and confirm row count equals the source count and Reservation_ID is unique.'),
P('Part II: Cleaning and feature engineering',H1),
P(f"The original is preserved as facility_reservations_original_copy.csv. Rows: {R['rows_raw']} raw to {R['rows_clean']} clean. Values were never invented: invalid values became missing, and cancelled reservations keep blank attendance (not applicable, not zero).")]
E_+=[tbl(dq,[3.2*inch,.6*inch,3.2*inch]),Spacer(1,6),
P('<b>Derived variables.</b> Lead_Time_Category (0-2 Last Minute, 3-7 Short Lead, 8-14 Medium Lead, >14 Early Booking); Reservation_Outcome (Used = Completed; Cancelled/No-Show = Not Completed; missing status = missing); Attendance_Ratio = Actual_Attendance / Participants, left blank when the denominator is zero or missing. Output: facility_reservations_clean.csv.'),
P('Part III: Exploratory data analysis',H1),
P(f"<b>Q1 Facility with most reservations:</b> Classroom, 100 of {R['n_all']} cleaned reservations ({100/R['n_all']*100:.1f}%)."),
img('chart1_facility_counts.png',4.6),
P(f"<b>Q2 Highest cancellation/no-show rate:</b> Individual Student, {uc[0]['Not Completed rate (%)']}% ({int(uc[0]['Not Completed'])}/{uc[0]['Reservations (known outcome)']}), versus {R['overall_nc']:.1f}% overall (denominator: {R['n_known']} reservations with known outcome). Faculty was lowest at {uc[3]['Not Completed rate (%)']}%."),
img('chart2_user_category_rate.png',4.6),
P(f"<b>Q3 Lead time vs outcome:</b> the not-completed rate rises with lead time: Last Minute {lt[0]['Not Completed rate (%)']}% ({lt[0]['Not Completed']}/{lt[0]['Total']}), Short {lt[1]['Not Completed rate (%)']}%, Medium {lt[2]['Not Completed rate (%)']}%, Early Booking {lt[3]['Not Completed rate (%)']}% ({lt[3]['Not Completed']}/{lt[3]['Total']}); denominator {R['n_lt']} reservations with known outcome and valid lead time. This is an association in this dataset, not proof of cause."),
img('chart3_leadtime_outcome.png',4.6),
P('Part IV: Predictive modelling',H1),
P(f"<b>Task 7.</b> Decision tree (max depth {m['depth']}, random_state=42, stratified 80/20 split: {m['n_train']} train / {m['n_test']} test) on {m['n_model']} reservations with a known outcome. Predictors known at submission: Facility_Type, User_Category, Time_Slot, Day_of_Week, Equipment_Requested (one-hot encoded) and Lead_Time_Days, Participants, Duration_Hours (median imputation fitted on training data only). <b>Leakage avoided</b> by excluding Usage_Status, Actual_Attendance, Cancellation_Days_Before and Attendance_Ratio, and by fitting imputers/encoders inside a pipeline on the training split only."),
tbl([['Metric (Not Completed = positive)','Value'],['Accuracy',f"{m['accuracy']:.3f}"],['Precision',f"{m['precision']:.3f}"],['Recall',f"{m['recall']:.3f}"],['F1',f"{m['f1']:.3f}"],['Majority-class baseline accuracy',f"{m['baseline']:.3f}"],['Confusion matrix (TN/FP/FN/TP)',f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}"],['5-fold CV F1 (mean +/- sd)',f"{R['cv']['f1_mean']:.2f} +/- {R['cv']['f1_std']:.2f}"]],[3.2*inch,1.8*inch],9),
Spacer(1,4),
P('<b>Top three predictors:</b> '+', '.join(f"{k} ({v:.2f})" for k,v in R['top3'] and [(k,imp[k]) for k in R['top3']])+'.'),
img('chart4_confusion_matrix.png',3.0),img('chart5_feature_importance.png',4.2),
P(f"<b>Task 8. Reliability: {R['verdict']}.</b> Test accuracy ({m['accuracy']:.2f}) is barely above the majority baseline ({m['baseline']:.2f}), recall is about half, and cross-validated scores are lower and unstable, so it should not drive decisions. <b>Limitation:</b> small, synthetic dataset and a small test set (n={m['n_test']}), so metrics vary a lot. <b>Improvement:</b> collect more real history, add informative submission-time features, tune with cross-validation and compare with a baseline and ensemble model. Undefined metrics use zero_division=0 and the split is stratified so both classes are in the test set."),
P('Part V: Requirements and use case model',H1)]
E_+=[tbl([fr[0]]+fr[1:],[2.4*inch,.8*inch,3.8*inch]),Spacer(1,6),
P('<b>Task 10.</b> Updated use case diagram (new analytical use cases in orange): View Reservation &amp; Cancellation Analytics, View Predicted Reservation Risk (include Generate Risk Prediction; extends Review Reservation Request), View Risk Explanation (extends View Predicted Reservation Risk), Send Reconfirmation Reminder (extends Submit Reservation Request for long-lead bookings). Include is used because a risk view always needs a prediction; extend is used where behaviour is optional or conditional.'),
img('UML_use_case_diagram.png',6.4),
P('Part VI: Decision support and critical analysis',H1),
img('dashboard_wireframe.png',6.6),
P('<b>Task 12.</b> <b>(a)</b> A high-risk prediction is a probability from a weak model, not a fact; rejecting automatically would be unfair to individuals, could wrongly block legitimate use and removes human judgement. It should trigger a reconfirmation or waitlist, with the administrator deciding. <b>(b)</b> Missing, duplicated, inconsistent or invalid data biases the learned patterns, shrinks usable training rows and produces unreliable rates and predictions; in the information system it causes wrong counts, double bookings and loss of trust in reports. Validation at entry prevents this. <b>(c)</b> Show the predicted risk with its label "predicted", the probability, the top contributing factors with their values (e.g. lead time, participants, user category), the observed historical rate for similar reservations with counts, data completeness/missing fields, model accuracy, and a suggested non-punitive action.'),
P('Recommendations',H1),
P('1) Add reconfirmation for early bookings (FR-DM-01). 2) Track cancellation/no-show rates per user category on the dashboard (FR-DM-02). 3) Introduce risk prediction only as advisory, after the model is retrained on real data and shows clear improvement over the baseline (FR-DM-03). 4) Add entry validation to stop invalid dates, counts and free-text categories.')]
SimpleDocTemplate('SAD_Lab_Exam_Report.pdf',pagesize=letter,leftMargin=.7*inch,rightMargin=.7*inch,topMargin=.6*inch,bottomMargin=.6*inch).build(E_)
