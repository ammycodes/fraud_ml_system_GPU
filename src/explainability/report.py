
class ReportGenerator:
    """Generates human-readable fraud model explanation"""

    def generate(self, metrics_df, path):

        best = metrics_df.sort_values("auc_pr", ascending=False).iloc[0]

        with open(path, "w") as f:

            f.write("# Fraud Model Report\n\n")

            f.write(f"Best Model: {best['model']}\n\n")

            f.write("## Why this model?\n")
            f.write(f"- Highest AUC-PR: {best['auc_pr']}\n")

            f.write("\n## Trade-offs\n")
            f.write("- Balances recall and precision\n")

            f.write("\n## Recommendation\n")
            f.write(f"- Use threshold: {best.get('threshold', 0.5)}\n")

            f.write("\n## Notes\n")
            f.write("- Monitor drift regularly\n")
