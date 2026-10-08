import os
import json
import pandas as pd


class LabelAuditor:
    """
    Audit the existing V1 weak complexity labels.

    IMPORTANT:
    This never modifies the original V1 label file.
    """

    def __init__(
        self,
        label_path=(
            "notebooks/data/evaluation/"
            "v1_complexity_labels.json"
        ),
        output_dir=(
            "notebooks/data/evaluation/"
            "label_audit"
        ),
    ):

        self.label_path = label_path
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    # ======================================================
    # LOAD LABELS
    # ======================================================

    def load_labels(self):

        print(
            f"Loading labels:\n"
            f"{self.label_path}"
        )

        with open(
            self.label_path,
            "r",
            encoding="utf-8",
        ) as f:

            labels = json.load(f)

        print(
            f"Loaded {len(labels)} examples."
        )

        return labels

    # ======================================================
    # BASIC AUDIT
    # ======================================================

    def audit(self, labels):

        df = pd.DataFrame(labels)

        print("\n" + "=" * 60)
        print("LABEL AUDIT")
        print("=" * 60)

        print(
            f"\nTotal examples: "
            f"{len(df)}"
        )

        # --------------------------------------------------
        # Complexity distribution
        # --------------------------------------------------

        print(
            "\nComplexity distribution:"
        )

        print(
            df["complexity"]
            .value_counts()
            .sort_index()
        )

        # --------------------------------------------------
        # Confidence statistics
        # --------------------------------------------------

        if "confidence" in df.columns:

            print(
                "\nConfidence statistics:"
            )

            print(
                df["confidence"]
                .describe()
            )

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------

        print(
            "\nMissing values:"
        )

        print(
            df.isnull().sum()
        )

        return df

    # ======================================================
    # CREATE REVIEW DATASET
    # ======================================================

    def create_review_file(
        self,
        df,
    ):

        review_df = df.copy()

        # --------------------------------------------------
        # Review fields
        # --------------------------------------------------

        review_df[
            "review_label"
        ] = review_df[
            "complexity"
        ]

        review_df[
            "review_status"
        ] = "pending"

        review_df[
            "review_notes"
        ] = ""

        # --------------------------------------------------
        # Confidence bucket
        # --------------------------------------------------

        if "confidence" in review_df.columns:

            def confidence_bucket(x):

                if x < 0.50:
                    return "VERY_LOW"

                elif x < 0.65:
                    return "LOW"

                elif x < 0.80:
                    return "MEDIUM"

                else:
                    return "HIGH"

            review_df[
                "confidence_bucket"
            ] = review_df[
                "confidence"
            ].apply(
                confidence_bucket
            )

        # --------------------------------------------------
        # Sort lowest confidence first
        # --------------------------------------------------

        if "confidence" in review_df.columns:

            review_df = (
                review_df
                .sort_values(
                    "confidence"
                )
                .reset_index(
                    drop=True
                )
            )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_path = os.path.join(
            self.output_dir,
            "label_review.csv",
        )

        review_df.to_csv(
            output_path,
            index=False,
        )

        print(
            "\nReview file created:"
        )

        print(output_path)

        return output_path

    # ======================================================
    # CREATE LOW-CONFIDENCE FILE
    # ======================================================

    def create_low_confidence_file(
        self,
        df,
        threshold=0.65,
    ):

        if "confidence" not in df.columns:

            raise ValueError(
                "No confidence column found."
            )

        low_confidence = (
            df[
                df["confidence"]
                < threshold
            ]
            .copy()
        )

        output_path = os.path.join(
            self.output_dir,
            "low_confidence_labels.csv",
        )

        low_confidence.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nLow-confidence examples: "
            f"{len(low_confidence)}"
        )

        print(
            f"Saved to:\n"
            f"{output_path}"
        )

        return low_confidence

    # ======================================================
    # SAVE CORRECTED LABELS
    # ======================================================

    def create_corrected_json(
        self,
        review_csv,
        output_path=(
            "notebooks/data/evaluation/"
            "v1_corrected_complexity_labels.json"
        ),
    ):

        review_df = pd.read_csv(
            review_csv
        )

        # --------------------------------------------------
        # Only finalized reviews
        # --------------------------------------------------

        if "review_status" not in review_df.columns:

            raise ValueError(
                "review_status column missing."
            )

        reviewed = review_df[
            review_df[
                "review_status"
            ].str.lower()
            == "reviewed"
        ].copy()

        print(
            f"\nReviewed examples: "
            f"{len(reviewed)}"
        )

        # --------------------------------------------------
        # Check labels
        # --------------------------------------------------

        valid_labels = {
            "simple",
            "medium",
            "complex",
        }

        invalid = reviewed[
            ~reviewed[
                "review_label"
            ].isin(valid_labels)
        ]

        if len(invalid) > 0:

            raise ValueError(
                "Invalid review labels found:\n"
                f"{invalid['review_label'].unique()}"
            )

        # --------------------------------------------------
        # Rebuild JSON
        # --------------------------------------------------

        corrected = []

        for _, row in reviewed.iterrows():

            item = {}

            # Preserve original fields
            for column in [
                "instruction",
                "complexity",
                "confidence",
            ]:

                if column in row:

                    value = row[column]

                    if pd.isna(value):
                        value = None

                    item[column] = value

            # Corrected label
            item[
                "complexity"
            ] = row[
                "review_label"
            ]

            # Review metadata
            item[
                "review_status"
            ] = "reviewed"

            item[
                "review_notes"
            ] = row.get(
                "review_notes",
                "",
            )

            corrected.append(
                item
            )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                corrected,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nCorrected label file:"
        )

        print(output_path)

        return output_path