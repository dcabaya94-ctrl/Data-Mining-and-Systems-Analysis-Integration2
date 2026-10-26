"""Stand-in synthetic dataset generator for the SAD laboratory examination.

The official facility_reservations.csv was not supplied, so this script builds a
synthetic file with the same intent as the instructor notes: ~400 transactions with
a manageable number of intentional missing values, duplicates, inconsistent
categories and invalid values. No real student data is used.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 420

facilities = ["Classroom", "Computer Lab", "Auditorium", "Gymnasium", "Conference Room", "Study Room"]
fac_p = [0.27, 0.18, 0.10, 0.14, 0.17, 0.14]
users = ["Student Organization", "Faculty", "Administrative Staff", "Individual Student"]
usr_p = [0.35, 0.30, 0.15, 0.20]
cap = {"Classroom": (15, 60), "Computer Lab": (15, 40), "Auditorium": (80, 300),
       "Gymnasium": (20, 120), "Conference Room": (6, 25), "Study Room": (2, 8)}

fac = rng.choice(facilities, N, p=fac_p)
usr = rng.choice(users, N, p=usr_p)
lead_band = rng.choice(4, N, p=[0.22, 0.30, 0.25, 0.23])
lead = np.array([rng.integers(0, 3) if b == 0 else rng.integers(3, 8) if b == 1
                 else rng.integers(8, 15) if b == 2 else rng.integers(15, 46) for b in lead_band])
book = pd.Timestamp("2025-08-01") + pd.to_timedelta(rng.integers(0, 120, N), unit="D")
res_date = book + pd.to_timedelta(lead, unit="D")
part = np.array([rng.integers(*cap[f]) for f in fac])
dur = rng.choice([1, 2, 3, 4, 5, 6], N, p=[0.2, 0.3, 0.22, 0.14, 0.08, 0.06])
slot = rng.choice(["Morning", "Afternoon", "Evening"], N, p=[0.38, 0.40, 0.22])
equip = rng.choice(["Yes", "No"], N, p=[0.4, 0.6])

# Not-completed probability depends on features known at submission time.
logit = np.full(N, -0.9)
logit += np.select([lead <= 2, lead <= 7, lead <= 14], [-0.7, -0.2, 0.2], 0.9)
logit += pd.Series(usr).map({"Individual Student": 0.7, "Student Organization": 0.3,
                             "Faculty": -0.5, "Administrative Staff": -0.2}).values
logit += pd.Series(fac).map({"Auditorium": 0.4, "Gymnasium": 0.3, "Study Room": 0.5,
                             "Classroom": -0.2, "Computer Lab": 0.0, "Conference Room": 0.0}).values
logit += np.where(slot == "Evening", 0.3, 0.0)
p_nc = 1 / (1 + np.exp(-logit))
not_done = rng.random(N) < p_nc
kind = rng.random(N) < 0.6  # among not-completed: 60% cancelled, 40% no-show
status = np.where(~not_done, "Used", np.where(kind, "Cancelled", "No-Show"))

attend = np.full(N, np.nan)
used = status == "Used"
attend[used] = np.round(part[used] * rng.uniform(0.55, 1.1, used.sum()))
attend[status == "No-Show"] = 0
cancel_days = np.full(N, np.nan)
for i in np.where(status == "Cancelled")[0]:
    cancel_days[i] = rng.integers(0, lead[i] + 1)

df = pd.DataFrame({
    "Reservation_ID": [f"R{i:04d}" for i in range(1, N + 1)],
    "Facility_Type": fac, "User_Category": usr,
    "Booking_Date": book.strftime("%Y-%m-%d"), "Reservation_Date": res_date.strftime("%Y-%m-%d"),
    "Day_of_Week": res_date.day_name(), "Time_Slot": slot, "Duration_Hours": dur,
    "Participants": part.astype(float), "Equipment_Requested": equip,
    "Usage_Status": status, "Actual_Attendance": attend, "Cancellation_Days_Before": cancel_days,
})

def pick(n, pool=None):
    pool = np.arange(N) if pool is None else pool
    return rng.choice(pool, n, replace=False)

# ---- intentional inconsistent categories ----
def vary(col, mapping, n):
    for i in pick(n):
        v = df.at[i, col]
        if v in mapping:
            df.at[i, col] = rng.choice(mapping[v])
vary("Facility_Type", {"Computer Lab": ["computer lab", "COMPUTER LAB", "Comp Lab"],
                       "Gymnasium": [" Gymnasium", "gym", "GYMNASIUM"],
                       "Conference Room": ["conference room", "Conf. Room"],
                       "Classroom": ["classroom ", "CLASSROOM"]}, 90)
vary("User_Category", {"Student Organization": ["student org", "Student Org.", "STUDENT ORGANIZATION"],
                       "Faculty": ["faculty ", "FACULTY"],
                       "Administrative Staff": ["Admin Staff", "admin staff"],
                       "Individual Student": ["student", "Individual student"]}, 80)
vary("Usage_Status", {"Used": ["used", "USED"], "No-Show": ["No Show", "no-show"],
                      "Cancelled": ["cancelled ", "Canceled"]}, 60)

# ---- intentional missing values ----
df.loc[pick(9), "Participants"] = np.nan
df.loc[pick(6), "Booking_Date"] = np.nan
df.loc[pick(7), "Time_Slot"] = np.nan
df.loc[pick(5), "Usage_Status"] = np.nan
df.loc[pick(8, np.where(used)[0]), "Actual_Attendance"] = np.nan

# ---- intentional invalid values ----
df.loc[pick(3), "Participants"] = -5
df.loc[pick(3), "Participants"] = 0
df.loc[pick(2), "Participants"] = 999
df.loc[pick(4), "Duration_Hours"] = rng.choice([0, 24], 4)
df.loc[pick(2), "Actual_Attendance"] = -3
for i in pick(4):  # booking date after reservation date
    df.at[i, "Booking_Date"] = (res_date[i] + pd.Timedelta(days=int(rng.integers(2, 10)))).strftime("%Y-%m-%d")

# ---- duplicate transactions: 14 exact copies + 2 copies differing only in case/spacing ----
dups = df.sample(14, random_state=7)
df = pd.concat([df, dups], ignore_index=True)
two = df.sample(2, random_state=11).copy()
two["Facility_Type"] = two["Facility_Type"].str.upper()
df = pd.concat([df, two], ignore_index=True)

df = df.sample(frac=1, random_state=3).reset_index(drop=True)
df.to_csv("facility_reservations.csv", index=False)
print(df.shape)
