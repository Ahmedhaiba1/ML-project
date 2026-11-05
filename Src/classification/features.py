"""Booking-time feature policy for the IsCanceled task.

Assumed prediction point: the moment the booking is created in the PMS.
"""
from src.data.loader import TARGET

# Not known at booking time, or directly/indirectly encode the outcome.
LEAKY_DROP = [
    "ReservationStatus",         # Canceled / Check-Out / No-Show: the target in disguise
    "ReservationStatusDate",     # date of the final status -> reveals the outcome
    "AssignedRoomType",          # assigned at/near check-in
    "BookingChanges",            # accumulates up to check-in or cancellation
    "RequiredCarParkingSpaces",  # >0 only for non-canceled bookings in H2 (verified in EDA)
]

# Timing is ambiguous: may be revised after booking. Excluded from the strict set,
# included in an ablation set to measure how much they change performance.
AMBIGUOUS = ["ADR", "TotalOfSpecialRequests"]


def feature_columns(df, include_ambiguous: bool = False):
    drop = set(LEAKY_DROP) | {TARGET}
    if not include_ambiguous:
        drop |= set(AMBIGUOUS)
    return [c for c in df.columns if c not in drop]
