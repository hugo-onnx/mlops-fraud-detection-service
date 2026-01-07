import time
import json
import pandas as pd
from io import BytesIO
from pathlib import Path

from fastapi.responses import FileResponse, StreamingResponse
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, UploadFile, File

from app.db.db import log_request, log_request_bulk
from app.config.config import settings, logger
from app.services.ml_service import model_service
from app.services.drift_service import drift_service
from app.models.schemas import Transaction, DriftReportRequest, PredictionResponse


router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024


# Prediction Endpoint
@router.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=200, 
    tags=["Prediction"]
)
def predict(
    transaction: Transaction, 
    background_tasks: BackgroundTasks, 
    request: Request
) -> PredictionResponse:
    """
    Receives a transaction and returns the predicted fraud probability.
    Logs the request asynchronously to the production database.
    """
    start_time = time.time()
    client_host = getattr(getattr(request, 'client', None), 'host', 'unknown')
    try:
        logger.info(f"Prediction request received from {client_host}")
        probability = model_service.predict(transaction.features)
        background_tasks.add_task(log_request, transaction.features, probability)
        
        latency = time.time() - start_time
        logger.info(json.dumps({
            "event": "prediction",
            "fraud_probability": probability,
            "latency_sec": round(latency, 4)
        }))
        
        return {
            "fraud_probability": probability, 
            "model_version": model_service.model_meta.get("version", "local")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during inference.")
    

@router.post(
    "/predict/batch", 
    tags=["Prediction"], 
    status_code=200
)
async def batch_predict_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None
):
    """
    Receives a CSV file with transaction features and returns a CSV file with an additional `fraud_probability` column.
    """
    start_time = time.time()
    client_host = getattr(getattr(request, 'client', None), 'host', 'unknown')

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .csv files are supported.")
    
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")

    try:
        logger.info(f"Batch prediction request from {client_host}")

        df = pd.read_csv(BytesIO(contents))
        required_features = model_service.feature_columns
        missing = [c for c in required_features if c not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        feature_df = df[required_features]

        predictions = []
        for _, row in feature_df.iterrows():
            features = row.to_dict()
            prob = model_service.predict(features)
            predictions.append(prob)
        predictions = pd.Series(predictions)

        df["fraud_probability"] = predictions

        if background_tasks:
            background_tasks.add_task(
                log_request_bulk,
                feature_df.to_dict(orient="records"),
                predictions.tolist()
            )

        output_buffer = BytesIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        latency = time.time() - start_time
        logger.info(json.dumps({
            "event": "batch_prediction",
            "rows": len(df),
            "latency_sec": round(latency, 4),
            "model_version": model_service.model_meta.get("version", "local")
        }))

        return StreamingResponse(
            output_buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=batch_predictions.csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error during batch inference."
        )
    

@router.get(
    "/predict/batch/template", 
    tags=["Prediction"], 
    status_code=200)
def download_batch_template():
    """
    Returns a CSV template with required feature columns and 10 example rows.
    """
    try:
        columns = model_service.feature_columns
        sample_rows = [
            [0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62],
            [0,1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.00898309914322813,0.0147241691924927,2.69],
            [1,-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204,378.66],
            [27219,-25.2663550194138,14.3232538097233,-26.8236729135114,6.34924780743689,-18.664250613469,-4.64740304866878,-17.9712120192706,16.6331030618556,-3.76835097141465,-8.303239351259,4.78325736701241,-6.6992520739678,0.846767864669643,-6.57627643636006,-0.0623303798952992,-5.96165987257541,-12.2184817176797,-4.79184198325342,0.894853521838799,1.65828884445902,1.78070097046593,-1.86131814726914,-1.18816729293127,0.156667050663465,1.76819198236914,-0.219916008250323,1.41185477685432,0.414656383638367,99.99],
            [2,-1.15823309349523,0.877736754848451,1.548717846511,0.403033933955121,-0.407193377311653,0.0959214624684256,0.592940745385545,-0.270532677192282,0.817739308235294,0.753074431976354,-0.822842877946363,0.53819555014995,1.3458515932154,-1.11966983471731,0.175121130008994,-0.451449182813529,-0.237033239362776,-0.0381947870352842,0.803486924960175,0.408542360392758,-0.00943069713232919,0.79827849458971,-0.137458079619063,0.141266983824769,-0.206009587619756,0.502292224181569,0.219422229513348,0.215153147499206,69.99],
            [2,-0.425965884412454,0.960523044882985,1.14110934232219,-0.168252079760302,0.42098688077219,-0.0297275516639742,0.476200948720027,0.260314333074874,-0.56867137571251,-0.371407196834471,1.34126198001957,0.359893837038039,-0.358090652573631,-0.137133700217612,0.517616806555742,0.401725895589603,-0.0581328233640131,0.0686531494425432,-0.0331937877876282,0.0849676720682049,-0.208253514656728,-0.559824796253248,-0.0263976679795373,-0.371426583174346,-0.232793816737034,0.105914779097957,0.253844224739337,0.0810802569229443,3.67],
            [7,-0.644269442348146,1.41796354547385,1.0743803763556,-0.492199018495015,0.948934094764157,0.428118462833089,1.12063135838353,-3.80786423873589,0.615374730667027,1.24937617815176,-0.619467796121913,0.291474353088705,1.75796421396042,-1.32386521970526,0.686132504394383,-0.0761269994382006,-1.2221273453247,-0.358221569869078,0.324504731321494,-0.156741852488285,1.94346533978412,-1.01545470979971,0.057503529867291,-0.649709005559993,-0.415266566234811,-0.0516342969262494,-1.20692108094258,-1.08533918832377,40.8],
            [8528,0.447395553302475,2.48195386638743,-5.66081393141405,4.4559228120932,-2.4437797540431,-2.18504026247234,-4.71614294470093,1.24980325173147,-0.718326066573691,-5.3903302556601,6.45418752494833,-8.48534657377678,0.635281408794591,-7.01990155916612,0.53981407134798,-4.6498642988132,-6.2883575087823,-1.33931232244731,2.26298478762517,0.549612970886705,0.756052550277073,0.140167768675,0.665411100673545,0.131463791724986,-1.90821741154788,0.334807598686864,0.748534284767756,0.175413794422896,1],
            [7,-0.89428608220282,0.286157196276544,-0.113192212729871,-0.271526130088604,2.6695986595986,3.72181806112751,0.370145127676916,0.851084443200905,-0.392047586798604,-0.410430432848439,-0.705116586646536,-0.110452261733098,-0.286253632470583,0.0743553603016731,-0.328783050303565,-0.210077268148783,-0.499767968800267,0.118764861004217,0.57032816746536,0.0527356691149697,-0.0734251001059225,-0.268091632235551,-0.204232669947878,1.0115918018785,0.373204680146282,-0.384157307702294,0.0117473564581996,0.14240432992147,93.2],
            [9,-0.33826175242575,1.11959337641566,1.04436655157316,-0.222187276738296,0.49936080649727,-0.24676110061991,0.651583206489972,0.0695385865186387,-0.736727316364109,-0.366845639206541,1.01761446783262,0.836389570307029,1.00684351373408,-0.443522816876142,0.150219101422635,0.739452777052119,-0.540979921943059,0.47667726004282,0.451772964394125,0.203711454727929,-0.246913936910008,-0.633752642406113,-0.12079408408185,-0.385049925313426,-0.0697330460416923,0.0941988339514961,0.246219304619926,0.0830756493473326,3.68],
        ]
        df = pd.DataFrame(sample_rows, columns=columns)

        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=batch_prediction_template.csv"
            }
        )

    except Exception as e:
        logger.exception(f"Error generating batch template: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate batch prediction template."
        )


# Monitoring Endpoints
@router.post("/monitoring/drift", tags=["Monitoring"])
def generate_drift_report(params: DriftReportRequest):
    """
    Triggers the generation of a data drift report using Evidently.
    """
    try:
        results = drift_service.generate_report(params.days)
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error generating drift report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate drift report: {str(e)}")
    

@router.get("/monitoring/drift/report", tags=["Monitoring"])
def get_drift_report():
    """Returns the last generated HTML data drift report."""
    report_path: Path = settings.REPORT_PATH

    logger.info(f"Checking for report at: {report_path.absolute()}")

    if not report_path.exists():
        logger.warning(f"Report file not found at {report_path.absolute()}")
        raise HTTPException(
            status_code=404,
            detail="No drift report found. Generate one first using POST /monitoring/drift"
        )
    
    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename="data_drift_report.html",
    )


# Health and Model Endpoints
@router.get("/health", tags=["System"])
def health_check():
    """Simple health check returning current model information."""
    if model_service.session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")
    
    return {
        "status": "ok",
        "model_name": model_service.model_meta.get("name"),
        "model_version": model_service.model_meta.get("version"),
        "model_description": model_service.model_meta.get("description"),
        "model_type": model_service.model_meta.get("type"),
    }


@router.post("/model/reload", tags=["System"])
def reload_model():
    """Triggers a reload of the champion model from MLflow registry."""
    try:
        model_service.load_model()
        return {
            "status": "success",
            "model_name": model_service.model_meta.get("name"),
            "model_version": model_service.model_meta.get("version"),
            "message": "Champion model reloaded successfully."
        }
    except Exception as e:
        logger.exception(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")