sap.ui.define([], function () {
	"use strict";
	return {

		name_family_status : function(name_family_status, sex){
			var mapdico = { 
				"Single / not married": "Célibataire", 
				"Married" : sex == "1" ? "Mariée": "Marié",
				"Civil marriage" : sex == "1" ? "Pacsée" : "Pacsé",
				"Widow" : sex == "1" ? "Veuve" : "Veuf",
				"Separated" : sex == "1" ? "Divorcée" : "Divorcé",
				"Unknown" : "Non renseigné",
			 }
			return mapdico[name_family_status] || "";
		},

		name_income_type : function(income_type, sex){
			var mapdico = { 
				'Working' : "Salaire d'employé du privé", 
				'State servant' : "Fonctionnaire", 
				'Commercial associate':  sex == "1" ? "Associée commerciale" : "Associé commercial",
				'Pensioner':  sex == "1" ? "Retraitée" : "Retraité",
				'Student':  sex == "1" ? "Étudiante" : "Étudiant",
				'Unemployed':  "Chômage",
				'Businessman':  sex == "1" ? "Femme d'affaires" : "Homme d'affaires",
				'Maternity leave' : "Congé maternité"
			 }
			return mapdico[income_type] || "";
		},


		occupation_type : function(income_type, sex){
			var mapdico = { 
				'Laborers' : "Ouvrier", 
				'Core staff' :"Personnel de base", 
				'Accountants' : "Comptable", 
				'Managers' : "Manager", 
				'Drivers' : sex == "1" ? "Camionneuse" : "Camionneur", 
				'Sales staff' : sex == "1" ? "Commerciale" : "Commercial", 
				'Cleaning staff' : "Agent de propreté", 
				'Cooking staff' : "Cuisinier",
				'Private service staff' : "Industrie du Service", 
				'Medicine staff' : "Secteur Médical", 
				'Security staff' : "Sécurité",
				'High skill tech staff' : "Ingénieur", 
				'Waiters/barmen staff' : sex == 1 ? "Serveuse de restaurant" : "Serveur de restaurant",
				'Low-skill Laborers' : "Ouvrier non qualifié", 
				'Realty agents' : "Secteur Immobilier", 
				'Secretaries' : "Secrétaire", 
				'IT staff' : "Technologies de l'Information (IT)",
				'HR staff': "Ressources Humaines (Suppôt de Satan)"
			 }
			return mapdico[income_type] || "Non renseigné";
		},

		name_contract_type: function(type){
		var mapdico = { 
			'Cash loans' : "Prêt d'argent", 
			'Consumer loans' :"Crédit à la consommation", 
			'Revolving loans' : "Carte de crédit", 
		}
		return mapdico[type] || "Non renseigné";
	}
	};
});