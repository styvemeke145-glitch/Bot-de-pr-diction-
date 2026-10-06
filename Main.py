import flet as ft
from scipy.stats import poisson

def main(page: ft.Page):
    page.title = "Bot Prédiction Live"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    # Champs de saisie
    v1_field = ft.TextField(label="Cote V1", value="15.1")
    x_field = ft.TextField(label="Cote Nul (X)", value="4.35")
    v2_field = ft.TextField(label="Cote V2", value="1.295")
    
    ligne_field = ft.TextField(label="Premier Over (ex: 3.5)", value="3.5")
    over_p_field = ft.TextField(label="Cote Over Premier", value="1.475")
    under_p_field = ft.TextField(label="Cote Under Premier", value="2.552")
    
    over_d_field = ft.TextField(label="Cote Over Dernier (Ligne + 1)", value="7.31")
    
    result_text = ft.Text(value="", size=18, weight=ft.FontWeight.BOLD, color=ft.colors.GREEN)

    def calculer_prediction(e):
        try:
            cote_v1 = float(v1_field.value)
            cote_x = float(x_field.value)
            cote_v2 = float(v2_field.value)
            ligne_premier = float(ligne_field.value)
            cote_over_premier = float(over_p_field.value)
            cote_over_dernier = float(over_d_field.value)
            
            ligne_dernier = ligne_premier + 1.0
            total_buts_actuels = max(0, int(ligne_premier - 0.5))
            
            inv_v1 = 1 / cote_v1
            inv_v2 = 1 / cote_v2
            somme_inv = inv_v1 + inv_v2 + (1 / cote_x)
            
            prob_1 = inv_v1 / somme_inv
            prob_2 = inv_v2 / somme_inv
            
            score_1 = round(total_buts_actuels * (prob_1 / (prob_1 + prob_2 + 1e-6)))
            score_2 = total_buts_actuels - score_1
            
            poids_p = 1 / cote_over_premier if cote_over_premier > 0 else 0.5
            poids_d = 1 / cote_over_dernier if cote_over_dernier > 0 else 0.1
            
            lambda_total = (ligne_premier * poids_p + ligne_dernier * poids_d) / (poids_p + poids_d)
            lambda_1 = lambda_total * (prob_1 / (prob_1 + prob_2 + 1e-6))
            lambda_2 = lambda_total * (prob_2 / (prob_1 + prob_2 + 1e-6))
            
            meilleure_prob = -1
            meilleur_add = (0, 0)
            
            for i in range(5):
                for j in range(5):
                    pi = poisson.pmf(i, max(0.1, lambda_1 - score_1 if score_1 < lambda_1 else 0.1))
                    pj = poisson.pmf(j, max(0.1, lambda_2 - score_2 if score_2 < lambda_2 else 0.1))
                    comb = pi * pj
                    if comb > meilleure_prob:
                        meilleure_prob = comb
                        meilleur_add = (i, j)
                        
            final_1 = score_1 + meilleur_add[0]
            final_2 = score_2 + meilleur_add[1]
            
            result_text.value = f"Score Actuel Estimé : {score_1} - {score_2}\nPrédiction Finale : {final_1} - {final_2}"
            page.update()
        except ValueError:
            result_text.value = "Erreur : Vérifiez vos valeurs numériques."
            page.update()

    btn = ft.ElevatedButton(text="Lancer la Prédiction", on_click=calculer_prediction)

    page.add(
        ft.Text("Bot Prédiction Live", size=24, weight=ft.FontWeight.BOLD),
        v1_field, x_field, v2_field,
        ligne_field, over_p_field, under_p_field, over_d_field,
        btn,
        result_text
    )

ft.app(target=main)
