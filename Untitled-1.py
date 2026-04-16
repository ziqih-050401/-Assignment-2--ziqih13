

#  Final interpretation 
print(f"""
Interpretation:
  • The AI tool {"INCREASES" if ate1 > 0 else "decreases"} throughput by ~{abs(ate1):.1f} tasks/shift (ATE = {ate1:+.2f}).
  • The AI tool {"INCREASES" if ate3 > 0 else "does NOT reduce"} error rates by ~{abs(ate3):.2f} ppt (ATE = {ate3:+.2f}).
  • This suggests the tool boosts QUANTITY but may come at a cost to QUALITY
    (a speed–accuracy trade-off), which is consistent with clerks processing
    more applications but making slightly more errors when the AI pre-fills
    fields that still require manual verification.
""")

#  Save 
df.to_csv("loan_operations_clean.csv", index=False)
print("Saved → loan_operations_clean.csv")
