import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Shared layout defaults (no flashy gradients, clean zinc) ────────────────
_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#a1a1aa', family='Inter', size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(
        bgcolor='#18181b',
        bordercolor='#27272a',
        font=dict(color='#fafafa', family='Inter', size=12)
    )
)

_AXIS = dict(
    gridcolor='#1c1c1f',
    zerolinecolor='#27272a',
    linecolor='#27272a',
    tickcolor='#3f3f46',
    tickfont=dict(color='#71717a', size=11)
)

# Clean 5-tone palette — Indigo → Sky → Emerald → Amber → Rose
# Subtle, professional, no neon
PALETTE = ['#6366f1', '#38bdf8', '#10b981', '#f59e0b', '#f43f5e']


def create_empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#71717a", family="Inter")
    )
    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=240
    )
    return fig


def create_score_gauge(score: float, title: str = "Overall Candidate Match") -> go.Figure:
    try:
        score = float(score) if score is not None else 0.0

        # Pick needle color by score band
        if score >= 75:
            bar_color = '#10b981'
        elif score >= 55:
            bar_color = '#f59e0b'
        else:
            bar_color = '#f43f5e'

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': title, 'font': {'size': 14, 'color': '#a1a1aa', 'family': 'Inter'}},
            number={'suffix': "%", 'font': {'size': 32, 'color': '#fafafa', 'family': 'Inter'}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#3f3f46",
                    'tickfont': {'color': '#71717a', 'size': 10}
                },
                'bar': {'color': bar_color, 'thickness': 0.28},
                'bgcolor': "#1c1c1f",
                'borderwidth': 1,
                'bordercolor': "#27272a",
                'steps': [
                    {'range': [0, 55],  'color': '#1c0a0a'},
                    {'range': [55, 75], 'color': '#1c1505'},
                    {'range': [75, 100], 'color': '#051c12'}
                ],
                'threshold': {
                    'line': {'color': "#10b981", 'width': 3},
                    'thickness': 0.7,
                    'value': 75
                }
            }
        ))

        fig.update_layout(
            **_LAYOUT,
            height=250
        )
        return fig
    except Exception as e:
        print(f"Error building score gauge: {e}")
        return create_empty_figure("Could not render gauge")


def create_radar_chart(categories: list, values: list, title: str = "Skill Category Profile") -> go.Figure:
    try:
        if not categories or not values or len(categories) != len(values):
            return create_empty_figure("Insufficient skill data")

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.12)',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=6, color='#6366f1'),
            name='Candidate'
        ))

        fig.update_layout(
            **_LAYOUT,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                    gridcolor='#1c1c1f',
                    linecolor='#27272a',
                    tickfont=dict(color='#71717a', size=10)
                ),
                angularaxis=dict(
                    gridcolor='#1c1c1f',
                    linecolor='#27272a',
                    tickfont=dict(color='#a1a1aa', size=11)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=dict(text=title, font=dict(size=14, color='#a1a1aa')),
            height=320
        )
        return fig
    except Exception as e:
        print(f"Error building radar chart: {e}")
        return create_empty_figure("Could not render radar chart")


def create_4d_candidate_radar_chart(candidate_results: list) -> go.Figure:
    try:
        if not candidate_results:
            return create_empty_figure("No candidate data")

        categories = ['Semantic Similarity', 'Hard Skill Match', 'Experience', 'Education Fit']

        fig = go.Figure()
        for idx, c in enumerate(candidate_results[:5]):
            semantic = round((c.get("sbert_score", 0) * 0.6 + c.get("tfidf_score", 0) * 0.4), 1)
            vals = [
                semantic,
                c.get("skill_score", 0),
                c.get("exp_score", 0),
                c.get("edu_score", 75)
            ]
            color = PALETTE[idx % len(PALETTE)]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor=f"rgba({r},{g},{b},0.1)",
                line=dict(color=color, width=1.8),
                name=c.get("filename", f"Candidate #{idx+1}")
            ))

        fig.update_layout(
            **_LAYOUT,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                    gridcolor='#1c1c1f',
                    linecolor='#27272a',
                    tickfont=dict(color='#71717a', size=10)
                ),
                angularaxis=dict(
                    gridcolor='#1c1c1f',
                    linecolor='#27272a',
                    tickfont=dict(color='#a1a1aa', size=11)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=dict(text="4-Dimension Candidate Evaluation", font=dict(size=14, color='#a1a1aa')),
            legend=dict(
                orientation="h", y=-0.2, x=0.5, xanchor="center",
                font=dict(color='#a1a1aa', size=11),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='#27272a', borderwidth=1
            ),
            height=360
        )
        return fig
    except Exception as e:
        print(f"Error building 4D radar: {e}")
        return create_empty_figure("Could not render radar chart")


def create_candidate_comparison_chart(candidate_results: list) -> go.Figure:
    try:
        if not candidate_results:
            return create_empty_figure("No candidate data to compare")

        filenames = [c.get("filename", "Candidate") for c in candidate_results]
        overall = [c.get("overall_score", 0) for c in candidate_results]
        sbert   = [c.get("sbert_score", 0)   for c in candidate_results]
        skills  = [c.get("skill_score", 0)   for c in candidate_results]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=filenames, x=overall, name='Overall Score',
                             orientation='h', marker_color='#6366f1'))
        fig.add_trace(go.Bar(y=filenames, x=sbert,   name='Semantic Match',
                             orientation='h', marker_color='#38bdf8'))
        fig.add_trace(go.Bar(y=filenames, x=skills,  name='Skill Match',
                             orientation='h', marker_color='#10b981'))

        fig.update_layout(
            **_LAYOUT,
            barmode='group',
            title=dict(text="Candidate Leaderboard", font=dict(size=14, color='#a1a1aa')),
            xaxis=dict(**_AXIS, title="Score (%)", range=[0, 100]),
            yaxis=dict(**_AXIS, autorange="reversed"),
            legend=dict(
                orientation="h", y=1.08, x=0, xanchor="left",
                font=dict(color='#a1a1aa', size=11),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=340
        )
        return fig
    except Exception as e:
        print(f"Error building comparison chart: {e}")
        return create_empty_figure("Could not render comparison chart")


def create_score_breakdown_bar(analysis_result: dict) -> go.Figure:
    try:
        if not analysis_result:
            return create_empty_figure("No analysis data")

        factors = ['Overall', 'Semantic', 'TF-IDF', 'Skills', 'Experience']
        scores  = [
            analysis_result.get('overall_score', 0),
            analysis_result.get('sbert_score', 0),
            analysis_result.get('tfidf_score', 0),
            analysis_result.get('skill_score', 0),
            analysis_result.get('exp_score', 0)
        ]

        fig = go.Figure(go.Bar(
            x=factors,
            y=scores,
            marker_color=PALETTE,
            text=[f"{s}%" for s in scores],
            textposition='outside',
            textfont=dict(color='#a1a1aa', size=11, family='Inter'),
            cliponaxis=False
        ))

        fig.update_layout(
            **_LAYOUT,
            title=dict(text="Score Breakdown", font=dict(size=14, color='#a1a1aa')),
            yaxis=dict(**_AXIS, range=[0, 115], title="Score (%)"),
            xaxis=dict(**_AXIS),
            height=270
        )
        return fig
    except Exception as e:
        print(f"Error building score breakdown bar: {e}")
        return create_empty_figure("Could not render score breakdown")


def create_skill_gap_chart(matched_skills: list, missing_skills: list) -> go.Figure:
    try:
        n_matched = len(matched_skills) if matched_skills else 0
        n_missing = len(missing_skills) if missing_skills else 0

        fig = go.Figure(go.Bar(
            x=['Matched Skills', 'Missing Skills'],
            y=[n_matched, n_missing],
            marker_color=['#10b981', '#f43f5e'],
            text=[n_matched, n_missing],
            textposition='outside',
            textfont=dict(color='#a1a1aa', size=12, family='Inter'),
            cliponaxis=False,
            width=[0.45, 0.45]
        ))

        fig.update_layout(
            **_LAYOUT,
            title=dict(text="Skill Gap Overview", font=dict(size=14, color='#a1a1aa')),
            yaxis=dict(**_AXIS, title="Count"),
            xaxis=dict(**_AXIS),
            height=250
        )
        return fig
    except Exception as e:
        print(f"Error building skill gap chart: {e}")
        return create_empty_figure("Could not render skill gap chart")


def create_interactive_taxonomy_chart(skill_db: dict, selected_category: str = None) -> go.Figure:
    try:
        if not skill_db:
            return create_empty_figure("No skill taxonomy loaded")

        df_cat = pd.DataFrame([
            {"Category": k.replace('_', ' ').title(), "Count": len(v), "raw_key": k}
            for k, v in skill_db.items()
        ])

        colors = [
            '#6366f1' if (selected_category and raw == selected_category) else '#27272a'
            for raw in df_cat["raw_key"]
        ]

        fig = go.Figure(go.Bar(
            x=df_cat["Category"],
            y=df_cat["Count"],
            marker_color=colors,
            text=df_cat["Count"],
            textposition='outside',
            textfont=dict(color='#71717a', size=11, family='Inter'),
            cliponaxis=False
        ))

        fig.update_layout(
            **_LAYOUT,
            title=dict(text="Skill Domain Distribution", font=dict(size=14, color='#a1a1aa')),
            xaxis=dict(**_AXIS, tickangle=-30),
            yaxis=dict(**_AXIS, title="Skill Count"),
            height=300
        )
        return fig
    except Exception as e:
        print(f"Error building taxonomy chart: {e}")
        return create_empty_figure("Could not render taxonomy chart")
