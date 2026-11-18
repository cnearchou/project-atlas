"""Project Atlas: a multi-layer quantitative equity forecasting and risk system.

The package follows the structure of the Project Atlas specification
(docs/Project_Atlas_Specification.docx). Each sub-package corresponds to one section:

    atlas.data          Section 0  Shared data infrastructure
    atlas.ingest        Technical Reference Part 2  Data sources and APIs
    atlas.fundamentals  Section 1  Fundamentals
    atlas.macro         Section 2  Macro
    atlas.news          Section 3  News, geopolitics, elections and laws
    atlas.flows         Section 4  Flows and positioning
    atlas.behaviour     Section 5  Behaviour and sentiment
    atlas.sectors       Section 6  Sector structure
    atlas.forecast      Section 7  Forecasting engine
    atlas.risk          Section 8  Risk management, execution and attribution
    atlas.research      Section 9  Hypothesis testing (and Section A hypotheses)
    atlas.pipeline      Section 10 Integration and daily schedule

Formulas are listed in docs/Project_Atlas_Technical_Reference.docx, Part 3. Comments throughout the
code cite both documents as "Spec §x.y" and "Ref 3.z".
"""

__version__ = "0.1.0"
