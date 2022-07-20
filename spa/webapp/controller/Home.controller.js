sap.ui.define([
	"sap/ui/core/mvc/Controller",
	"../model/formatter",
	"sap/m/MessageBox",
	"sap/m/MessageToast",
	"sap/ui/model/json/JSONModel",
	"sap/ui/core/BusyIndicator",
	'sap/ui/core/Fragment',
	"sap/m/MenuItem",
	"sap/m/Dialog",
	"sap/m/Button",
	"sap/m/Image"
], function(Controller, formatter, MessageBox, MessageToast,  JSONModel, BusyIndicator, Fragment, MenuItem, Dialog, Button, Image) {
	"use strict";

	return Controller.extend("cna.projet7.controller.App", {

		formatter: formatter,

		onInit: function () {
			var oData = {
				text: {
				  line1: "line1",
				  line2: "line2",
				  line3: "line3",
				  line4: "line4"
				}
			  };
			  var oModel = new JSONModel(oData);
			  this.getView().setModel(oModel,"bing");

			  // Préparons un modèle JSON pour les données filtre
			  this.getView().setModel(new JSONModel({loan_id:100001}),"filter");
			  
			  this.getView().setModel(new JSONModel({ first_search_not_already: true }),"page_config");

			  var insuranceLogo = sap.ui.require.toUrl("cna/projet7/img/insurance.png");
			  var ocrLogo = sap.ui.require.toUrl("cna/projet7/img/ocr.png");
			  this.getView().setModel(new JSONModel( {insurance : insuranceLogo, ocr: ocrLogo}  ), "img");			  

			  /* Modèle JSON des images d'explicabilité */
			  var shap_summary = sap.ui.require.toUrl("cna/projet7/model/img/shap.png");
			  let oExplicabilityJSONModel = new JSONModel( 
				{ shap_summary : shap_summary,
				  shap_force   : "",
				  lime_values  : [],
				  box_plot_url : "" } );
			  this.getView().setModel(oExplicabilityJSONModel, "explicability_img");

			  jQuery.sap.delayedCall(0, this, function () {this.getView().byId("input_id_contrat").focus(); }); 

		},

		getScoreColor: function(){
			customer_score = this.getView().getModel(loan_data).getData("/customer_score");

			if ( customer_score > 80 ){
				return "Good";
			}
			if ( customer_score > 50 ){
				return "Critical";
			} 
			else {
				return "Error";
			} 
		},

		onOpenDialog: function(){
			let header_text = "Dashboard de scoring crédit pour calculer la probabilité qu’un client rembourse son crédit.";
			let details = "<p>Ce dashboard a été réalisé par <strong>Christophe Naciri</strong> dans le cadre d'une formation en Data Science avec <a href='https://openclassrooms.com/fr/'>Openclassrooms</a>.</p>"+
			"<p>Il est bâti sur la stack technique suivante :</p>"+
			"<ul>"+
			"<li>FastAPI avec Python 3.10</li>"+
			"<li>SAPUI5 pour le framework HTML5</li>"+
			"<li>Scikit-learn, Pandas, Numpy pour la partie Machine Learning</li>"+
			"</ul>";
			MessageBox.information(header_text, 
			{	details : details,
				styleClass: "sapUiResponsivePadding--header sapUiResponsivePadding--content sapUiResponsivePadding--footer"
			});
		},

		onSearch: function( ){

			var oView = this.getView();
			
			let loan_id = oView.getModel("filter").getData("/filter").loan_id
			let oJSONModel = new JSONModel();

			let source_path = window.location.href.split("webapp")[0]+"loans/" + loan_id;
			BusyIndicator.show()

			/* Obtenir les données de prédiction + détails d'un client */
			oJSONModel.loadData( source_path ).then( function(d1,d2){
				var oReceivedData = oJSONModel.getData();
				 if (oReceivedData.hasOwnProperty("error")){
					// Une erreur a été détectée
					MessageToast.show("Ce numéro de demande n'a pas été trouvé dans la base de données.")
				 }
				 else {
					// All good
					let oFormViewModel = oView.getModel("loan_data");
					oFormViewModel.setData( oReceivedData  );

					oView.getModel("page_config").setProperty("/first_search_not_already",false);
				}


			} ).catch( function(oError)
			{
				console.log("Erreur AJAX dans onSearch() de Home.controller.js");
			}   ).finally (function() { BusyIndicator.hide();} )

			/* Obtenir les données d'explication LIME */
			let oLIMEJSONModel = new JSONModel();
			var oInteractiveBarChart =  oView.byId("limeInteractiveBarChart");  
			var oImageBoxPlot =  oView.byId("imgBoxPlot");  
			oInteractiveBarChart.setBusy(true);  oImageBoxPlot.setVisible(false);  
			oLIMEJSONModel.loadData( window.location.href.split("webapp")[0]+"lime_tabular_explain/" + loan_id ).then( function(d1,d2){
				oInteractiveBarChart.setBusy(false);
				oImageBoxPlot.setVisible(true);
				var oReceivedData = oLIMEJSONModel.getData();
				
				var arrays_of_2arr = JSON.parse(oReceivedData);
				var arrays_of_objects = arrays_of_2arr.map( x => ({name : x[0], value : x[1] }) );
				var arrays_of_objects = arrays_of_2arr.map( x => ({name : x[0], value : parseInt(parseFloat(String(x[1] * 100)).toFixed(0)) }) );
				oInteractiveBarChart.setDisplayedBars(arrays_of_objects.length);	

				var oItemTemplate = new sap.suite.ui.microchart.InteractiveBarChartBar({label:"{explicability_img>name}", value: "{explicability_img>value}" });
				oView.getModel("explicability_img").setProperty("/lime_values",arrays_of_objects);
				oView.getModel("explicability_img").refresh(true);
				//oInteractiveBarChart.bindBars("explicability_img>/lime_values", oItemTemplate);
			});		
			
			var oExplicabilityJSONModel = oView.getModel("explicability_img");
			oExplicabilityJSONModel.setProperty("/box_plot_url",  window.location.href.split("webapp")[0]+"boxplot/" + loan_id );

		},

		onRandomIdSearch: function(){
			let source_path = window.location.href.split("webapp")[0]+"random_loan_id/";
			var oView = this.getView();
			let oJSONModel = new JSONModel();
			oJSONModel.loadData( source_path ).then( function(d1,d2){
				var oReceivedData = oJSONModel.getData();
				oView.getModel("filter").setProperty("/loan_id", oReceivedData ); 
			} );

		},

		scoreValueColor: function( score){
			if (score > 85){return "Good"}
			return "Error"
		},

		formatPaymentRate: function( rate ){
			if (rate){
				let rate_float = parseFloat(rate.replace(',','.'));
				rate_float = rate_float * 100; 
				return Math.round(rate_float  * 100) / 100;
			}
		},

		onExplicabilityPress: function () {
			var oView = this.getView(),
				oButton = oView.byId("explication_button");

			if (!this._oMenuFragment) {
				this._oMenuFragment = Fragment.load({
					id: oView.getId(),
					name: "cna.projet7.view.fragment.menu",
					controller: this
				}).then(function(oMenu) {
					oMenu.openBy(oButton);
					this._oMenuFragment = oMenu;
					return this._oMenuFragment;
				}.bind(this));
			} else {
				this._oMenuFragment.openBy(oButton);
			}
		},

		onExplicabilityMenuAction: function(oEvent) {
			var oItem = oEvent.getParameter("item"),
				sItemPath = "";

			while (oItem instanceof MenuItem) {
				sItemPath = oItem.getText() + " > " + sItemPath;
				oItem = oItem.getParent();
			}

			sItemPath = sItemPath.substr(0, sItemPath.lastIndexOf(" > "));

			this.imageDialogPress(sItemPath);
			//MessageToast.show("Action triggered on item: " + sItemPath);
		},

		imageDialogPress: function (sItemPath) {
			var sPath = ""; var oView = this.getView();
			var oExplicabilityJSONModel = this.getView().getModel("explicability_img");
			switch (sItemPath){
				case "Résumé SHAP":
					sPath = "explicability_img>/shap_summary";
				break;
				case "SHAP Force":
					sPath = "explicability_img>/shap_force";
					oExplicabilityJSONModel.setProperty("/shap_force",  window.location.href.split("webapp")[0]+"shap_force/" + 10001 );
					
					/*
					let oJSONModel = new JSONModel();
					oJSONModel.loadData( source_path ).then( function(d1,d2){
						var oReceivedData = oJSONModel.getData();
						oExplicabilityJSONModel.setProperty("/shap_force",  oReceivedData );
					});*/
					break;
			}

			if (!this.oImageDialog) {
				this.oImageDialog = new Dialog({
					title: "Résumé SHAP",
					content: new Image({
						id: "imageId",
						src: {
							path: sPath 
						}
					}),
					endButton: new Button({
						text: "Close",
						press: function () {
							this.oImageDialog.close();
						}.bind(this)
					})
				});

				// to get access to the controller's model
				this.getView().addDependent(this.oImageDialog);
			}
			else {
				this.oImageDialog.getContent()[0].bindProperty("src",sPath)
				//this.oImageDialog.getContent()[0].setSrc(sPath);
			}

			this.oImageDialog.open();
		},		

	});
});