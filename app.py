import streamlit as st
import os, re, warnings, requests, base64, json
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Configuración página ─────────────────────────────────
st.set_page_config(page_title="Clasificador Fallas CEC",page_icon="⚡",layout="wide")
st.markdown("""<style>
.resultado-FM{border-left:5px solid #e74c3c;background:#fdf0ef;border-radius:8px;padding:14px;margin:10px 0}
.resultado-E{border-left:5px solid #f39c12;background:#fef9ef;border-radius:8px;padding:14px;margin:10px 0}
.resultado-I{border-left:5px solid #27ae60;background:#edfaf1;border-radius:8px;padding:14px;margin:10px 0}
.badge-FM{background:#e74c3c;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.badge-E{background:#f39c12;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.badge-I{background:#27ae60;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.geo-card{background:#eaf4fb;border:2px solid #3498db;border-radius:8px;padding:14px;margin:10px 0}
.sec{font-size:16px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-bottom:10px}
</style>""",unsafe_allow_html=True)

# ─── Configuración GitHub ─────────────────────────────────
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
ARCHIVO_CSV  = "fallas_ingresadas.csv"
EXCEL_PATH   = "Estadisticas_de_Interrupciones.xlsx"

# ─── Constantes ───────────────────────────────────────────
ETIQUETAS    = {"FM":"Fuerza Mayor","E":"Externa","I":"Interna"}
COLORES      = {"FM":"#e74c3c","E":"#f39c12","I":"#27ae60"}
COLORES_ALIM = {2:"blue",6:"red",7:"green",8:"purple",9:"orange",10:"darkblue"}

# ─── Comunas disponibles ──────────────────────────────
COMUNAS = ["Curicó","Teno","Molina","Romeral","Chimbarongo","Otra"]
TENSIONES = ["MT","BT"]

CUADRILLAS = [
    "Cuadrilla 1 — Curicó Norte","Cuadrilla 2 — Curicó Sur",
    "Cuadrilla 3 — Teno","Cuadrilla 4 — Molina",
    "Cuadrilla 5 — Romeral","Cuadrilla 6 — Chimbarongo",
    "Cuadrilla 7 — Emergencia","Otra (escribir abajo)",
]

CAUSAS = {
    "Árboles / Vegetación":{"cal":"I","subs":[
        ("Caída de árbol sobre la red","Caída de árbol sobre red MT/BT"),
        ("Caída de gancho o rama","Caída de rama o gancho sobre conductor"),
        ("Poda inadecuada por cliente","Cliente efectuó poda propia indebida afectando red"),
        ("Poda inadecuada por municipalidad","Municipalidad efectuó poda que dañó la red"),
        ("Sector sin podar","Sector sin podar o mal podado afecta red"),
        ("Daño por empresa forestal","Daño por faena de empresa forestal en franja"),
        ("Otro árbol/vegetación","Daño por vegetación — causa no especificada")]},
    "Condiciones Atmosféricas":{"cal":"I","subs":[
        ("Temporal / viento fuerte","Interrupción por temporal o viento fuerte"),
        ("Lluvia intensa","Interrupción por lluvia intensa"),
        ("Granizo","Interrupción por granizo"),
        ("Nieve","Interrupción por nieve"),
        ("Hielo en conductores","Interrupción por hielo en conductores"),
        ("Rayo / descarga eléctrica","Descarga eléctrica (rayo) impacta instalación"),
        ("Neblina / ambiente salino","Interrupción por neblina o ambiente salino"),
        ("Temperatura extrema","Interrupción por temperatura extrema")]},
    "Accidentes / Vehículos":{"cal":"FM","subs":[
        ("Choque de vehículo a poste","Choque de vehículo a poste de distribución"),
        ("Vehículo alto bota cable MT","Vehículo > 4.5m bota cable media tensión"),
        ("Vehículo alto bota cable BT","Vehículo > 4.5m bota cable baja tensión"),
        ("Choque a tirante / retenida","Choque de vehículo a tirante o retenida"),
        ("Máquina retroexcavadora","Daño por máquina retroexcavadora"),
        ("Daño en faena particular","Daño debido a faena en propiedad particular"),
        ("Otro accidente","Accidente — causa no especificada")]},
    "Incendio":{"cal":"FM","subs":[
        ("Calor por incendio externo","Calor en instalación por incendio externo"),
        ("Quema de pastizales","Interrupción por quema de pastizales"),
        ("Solicitud de bomberos","Intervención a solicitud de bomberos"),
        ("Otro incendio","Incendio — causa no especificada")]},
    "Eventos Catastróficos":{"cal":"FM","subs":[
        ("Inundación","Interrupción por inundación"),
        ("Aluvión / deslizamiento","Interrupción por aluvión o deslizamiento"),
        ("Movimiento telúrico (sismo)","Interrupción por sismo"),
        ("Otro catastrófico","Evento catastrófico — no especificado")]},
    "Animales":{"cal":"I","subs":[
        ("Ave en instalación","Contacto de ave con instalación eléctrica"),
        ("Mamífero en instalación","Contacto de mamífero con instalación"),
        ("Roedor daña cable","Roedor daña aislación de cable"),
        ("Insecto / colmena","Insecto o colmena en instalación"),
        ("Otro animal","Animal — causa no especificada")]},
    "Falla en la Red (Interna)":{"cal":"I","subs":[
        ("Operación imprevista de equipo","Operación imprevista de equipo de maniobra"),
        ("Error de operación","Error de operación por personal"),
        ("Falla de material / equipo","Falla de material o equipo de la red"),
        ("Envejecimiento de materiales","Falla por envejecimiento de materiales"),
        ("Pérdida de aislación","Pérdida de aislación en conductor o equipo"),
        ("Sobrecarga / desequilibrio","Interrupción por sobrecarga"),
        ("Falla transformador distribución","Falla en transformador de distribución"),
        ("Mantenimiento preventivo","Corte programado por mantenimiento"),
        ("Otra falla interna","Falla interna — no especificada")]},
    "Causa Externa (Sistema)":{"cal":"E","subs":[
        ("Blackout del sistema","Blackout del sistema eléctrico nacional"),
        ("Falla en transmisión (Transelec)","Falla en línea/subestación de transmisión"),
        ("Falla en subtransmisión","Falla en línea/subestación de subtransmisión"),
        ("Racionamiento con decreto","Racionamiento energético con decreto vigente"),
        ("Baja/sobre frecuencia","Protección por baja o sobre frecuencia"),
        ("Programada generación","Interrupción programada por generación"),
        ("Otra causa externa","Causa externa — no especificada")]},
    "Actos Vandálicos":{"cal":"FM","subs":[
        ("Robo de conductor","Robo de conductor en red"),
        ("Robo de equipo","Robo de equipo eléctrico"),
        ("Objeto lanzado","Objeto lanzado sobre red"),
        ("Atentado / explosivos","Atentado con explosivos"),
        ("Sabotaje","Sabotaje a instalación eléctrica"),
        ("Otro vandálico","Acto vandálico — no especificado")]},
    "Instalación de Cliente":{"cal":"I","subs":[
        ("Falla en artefacto eléctrico","Falla en artefacto eléctrico del cliente"),
        ("Problema neutro / puesta tierra","Problema de neutro o puesta a tierra"),
        ("Capacidad insuficiente S/E","Capacidad insuficiente subestación cliente"),
        ("Desconexión solicitada","Desconexión a solicitud del cliente"),
        ("Otro cliente","Instalación cliente — no especificada")]},
}

# ─── Funciones GitHub ─────────────────────────────────────
def cargar_hist():
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{ARCHIVO_CSV}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r       = requests.get(url, headers=headers)
    if r.status_code == 200:
        contenido = base64.b64decode(r.json()["content"]).decode("utf-8")
        return pd.read_csv(StringIO(contenido))
    return pd.DataFrame()

def guardar(fila, foto_bytes=None, foto_nombre=None):
    dh  = cargar_hist()
    n   = len(dh) + 1
    fila["N°"]         = n
    fila["Fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fila["Foto"]       = f"fotos/registro_{n}.jpg" if foto_bytes else ""
    dh  = pd.concat([dh, pd.DataFrame([fila])], ignore_index=True)
    csv_b64 = base64.b64encode(dh.to_csv(index=False).encode()).decode()
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{ARCHIVO_CSV}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}","Content-Type":"application/json"}
    r       = requests.get(url, headers=headers)
    sha     = r.json().get("sha") if r.status_code == 200 else None
    payload = {"message":f"Registro N°{n} — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               "content":csv_b64}
    if sha: payload["sha"] = sha
    requests.put(url, headers=headers, data=json.dumps(payload))
    if foto_bytes:
        foto_b64 = base64.b64encode(foto_bytes).decode()
        url_foto = f"https://api.github.com/repos/{GITHUB_REPO}/contents/fotos/registro_{n}.jpg"
        r_foto   = requests.get(url_foto, headers=headers)
        sha_foto = r_foto.json().get("sha") if r_foto.status_code == 200 else None
        payload_foto = {"message":f"Foto registro N°{n}","content":foto_b64}
        if sha_foto: payload_foto["sha"] = sha_foto
        requests.put(url_foto, headers=headers, data=json.dumps(payload_foto))
    return n

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    col_l1,col_l2=st.columns(2)
    with col_l1:
        if os.path.exists("logo_udec.png"):
            st.image("logo_udec.png",width=100)
    with col_l2:
        if os.path.exists("logo_cec.jpg"):
            st.image("logo_cec.png",width=100)
    st.markdown("---")
    st.markdown("# CEC Fallas")
    st.markdown("[Sitio web CEC](https://cecltda.cl/)")
    st.markdown("---")
    if not os.path.exists(EXCEL_PATH):
        st.error("Excel no encontrado"); st.stop()
    else:
        st.success("Sistema conectado")
    st.markdown("---")
    pagina=st.radio("Navegación:",[
        "Registrar Falla","Historial",
        "Datos Históricos","Datos Actuales",
        "Mapa de Calor","Mapa General","Ayuda"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:15px;color:#888;text-align:center;padding:10px 0'>
        <b>Desarrollado por:</b><br><br>
        <a href='https://www.linkedin.com/in/gustavo-puentes-lermanda-78830a25a'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           Gustavo Puentes Lermanda</a><br><br>
        <a href='https://www.linkedin.com/in/sofia-eliana-nahuelpán-álvarez-62b11a351'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           Sofía Nahuelpán Álvarez</a><br><br>
        <a href='https://www.linkedin.com/in/PERFIL-SEBASTIAN'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           Sebastián Castro Astudillo</a><br><br>
        <i>Depto. Ingeniería Eléctrica<br>Universidad de Concepción</i>
    </div>
    """,unsafe_allow_html=True)
    st.caption("Clasificador CEC v3 © 2026")

# ─── Cargar modelos ───────────────────────────────────────
@st.cache_resource(show_spinner="Cargando datos y entrenando modelos...")
def cargar_modelos(path):
    def tf(v):
        try: return float(str(v).replace(",","."))
        except: return np.nan
    xl  = pd.read_excel(path,sheet_name=None)
    raw = xl["BD"].iloc[1:].copy()
    raw.columns=["_","INC","Alimentador","Trafos","KVA","Clientes","Inicio","Fin",
                 "Duracion","Causa","Calificacion","Comuna_id","Comuna","Tension",
                 "X","Y","Descripcion","Validacion","año","Falta_coord"]
    raw["X_f"]=raw["X"].apply(tf); raw["Y_f"]=raw["Y"].apply(tf)
    raw.loc[(raw["X_f"]<280000)|(raw["X_f"]>380000),"X_f"]=np.nan
    raw.loc[(raw["Y_f"]<6080000)|(raw["Y_f"]>6160000),"Y_f"]=np.nan
    df=raw[["INC","Alimentador","Calificacion","Descripcion","año","X_f","Y_f","Comuna"]].copy()
    df["Calificacion"]=df["Calificacion"].str.strip().str.upper()
    df=df[df["Calificacion"].isin(["I","FM","E"])]
    u2w=Transformer.from_crs("EPSG:32719","EPSG:4326",always_xy=True)
    def u2l(x,y): lo,la=u2w.transform(x,y); return la,lo
    dg=df.dropna(subset=["X_f","Y_f","Alimentador"]).copy()
    dg["Alimentador"]=dg["Alimentador"].astype(int)
    knn=KNeighborsClassifier(n_neighbors=5,weights="distance")
    knn.fit(dg[["X_f","Y_f"]].values,dg["Alimentador"].values)
    info={}
    for alim,grp in dg.groupby("Alimentador"):
        xc,yc=grp["X_f"].mean(),grp["Y_f"].mean(); la,lo=u2l(xc,yc)
        pts=[u2l(r.X_f,r.Y_f) for _,r in grp.iterrows()]
        info[int(alim)]={"x_utm":xc,"y_utm":yc,"lat":la,"lon":lo,
            "comunas":", ".join(sorted(set(grp["Comuna"].dropna().astype(str)))),
            "n_fallas":len(grp),"pts":pts}
    STOP={"de","la","el","en","y","a","los","las","por","con","se","del","al",
          "un","una","es","que","no","le","su","sus","para","como","fue",
          "pero","ha","hay","lo","me","si","o","e","u"}
    def limpiar(t):
        if not isinstance(t,str): return ""
        t=t.lower()
        for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
            t=t.replace(a,b)
        t=re.sub(r"[^a-z0-9\s]"," ",t)
        return " ".join(x for x in t.split() if x not in STOP and len(x)>2)
    dm=df.dropna(subset=["Descripcion","Calificacion"]).copy()
    dm["txt"]=dm["Descripcion"].apply(limpiar)
    modelo=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=5000,min_df=2)),
        ("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
    modelo.fit(dm["txt"],dm["Calificacion"])
    return modelo,knn,info,limpiar,u2w

modelo,knn_alim,info_alim,limpiar,u2w=cargar_modelos(EXCEL_PATH)
def u2l(x,y): lo,la=u2w.transform(x,y); return la,lo

# ─── Funciones core ───────────────────────────────────────
def predecir(desc):
    tl=limpiar(desc); pm=modelo.predict([tl])[0]
    pr=dict(zip(modelo.classes_,modelo.predict_proba([tl])[0]))
    return {"pred":pm,"conf":round(pr[pm]*100,1),
            "FM":round(pr.get("FM",0)*100,1),
            "E":round(pr.get("E",0)*100,1),
            "I":round(pr.get("I",0)*100,1)}

def id_alim(x,y):
    if not x or not y or x<1000: return None
    p=np.array([[x,y]]); alim=int(knn_alim.predict(p)[0])
    proba=dict(zip(knn_alim.classes_,knn_alim.predict_proba(p)[0]))
    dist=float(knn_alim.kneighbors(p,n_neighbors=1)[0][0][0])
    c=info_alim[alim]; dc=np.sqrt((x-c["x_utm"])**2+(y-c["y_utm"])**2)
    return {"alim":alim,"conf":round(proba[alim]*100,1),"dist_p":round(dist,1),
            "dist_c":round(dc,1),"comunas":c["comunas"],
            "votos":{int(a):round(p*100,1) for a,p in proba.items() if p>0.01}}

# ══════════════════════════════════════════════════════════
# PÁGINA 1 — REGISTRAR FALLA
# ══════════════════════════════════════════════════════════
if pagina=="Registrar Falla":
    st.markdown("## Registro de Interrupción")
    st.markdown('<div class="sec">Sección 1 — Operador</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: usuario=st.text_input("Nombre del operador *",placeholder="Ej: Juan Pérez")
    with c2: cuad_sel=st.selectbox("Cuadrilla *",CUADRILLAS)
    with c3:
        if "Otra" in cuad_sel:
            cuadrilla=st.text_input("Nombre de cuadrilla:")
        else:
            cuadrilla=cuad_sel
            st.text_input("Cuadrilla:",value=cuadrilla,disabled=True)
    st.markdown("---")
    st.markdown('<div class="sec">Sección 2 — Descripción de la falla</div>',unsafe_allow_html=True)
    modo=st.radio("Modo:",["Guiado (recomendado)","Manual"],horizontal=True)
    desc=""; cat=""; sub=""; cal_sug=None
    if "Guiado" in modo:
        ca,cb=st.columns(2)
        with ca: categoria=st.selectbox("Categoría:",list(CAUSAS.keys()))
        with cb:
            subs_opts=[s[0] for s in CAUSAS[categoria]["subs"]]
            subcausa=st.selectbox("Subcausa:",subs_opts)
        base=next((d for n,d in CAUSAS[categoria]["subs"] if n==subcausa),subcausa)
        det=st.text_area("Detalles adicionales:",placeholder="Sector, equipo, observaciones...",height=70)
        desc=base+(f". {det}" if det.strip() else "")
        cat=categoria; sub=subcausa; cal_sug=CAUSAS[categoria]["cal"]
        st.info(f"Descripción generada: {desc}")
    else:
        desc=st.text_area("Descripción:",placeholder="Ej: Se abre reconectador RCU5724...",height=100)
        cat="Manual"; sub="Manual"
    st.markdown("---")

    # Sección 3: Datos técnicos
    st.markdown('<div class="sec">Sección 3 — Datos técnicos del evento</div>',unsafe_allow_html=True)
    COMUNA_IDS = {
        "Curicó":"7301","Teno":"7308","Molina":"7304",
        "Romeral":"7306","Chimbarongo":"6102","Otra":"",
    }
    d1,d2,d3 = st.columns(3)
    with d1:
        trafos = st.number_input("Trafos afectados:",min_value=0,value=0,step=1)
        kva    = st.number_input("KVA:",min_value=0.0,value=0.0,step=0.5,format="%.1f")
    with d2:
        clientes   = st.number_input("Clientes afectados:",min_value=0,value=0,step=1)
        comuna_sel = st.selectbox("Comuna:",COMUNAS)
    with d3:
        tension_sel = st.selectbox("Tipo de tensión:",TENSIONES)
        comuna_id   = COMUNA_IDS.get(comuna_sel,"")
        st.text_input("Comuna ID:",value=comuna_id,disabled=True)

    st.markdown("---")

    # Sección 4: Fecha/hora y duración
    st.markdown('<div class="sec">Sección 4 — Fechas y duración</div>',unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1:
        fecha_inicio = st.date_input("Fecha inicio:")
        hora_inicio  = st.time_input("Hora inicio:",step=60)
    with f2:
        fecha_fin = st.date_input("Fecha fin:")
        hora_fin  = st.time_input("Hora fin:",step=60)
    with f3:
        dt_ini = datetime.combine(fecha_inicio, hora_inicio)
        dt_fin = datetime.combine(fecha_fin, hora_fin)
        dur_min = max(0, (dt_fin - dt_ini).total_seconds() / 60)
        dur_hrs = dur_min / 60
        st.metric("Duración calculada", f"{dur_hrs:.2f} hrs")
        st.caption(f"= {dur_min:.0f} minutos")

    st.markdown("---")

    # Sección 5: Foto
    st.markdown('<div class="sec">Sección 5 — Foto de la falla (opcional)</div>',unsafe_allow_html=True)
    foto = st.file_uploader("Foto de la falla:",
                            type=["jpg","jpeg","png"],
                            help="Puedes tomar la foto desde el celular y subirla aquí")
    if foto:
        st.image(foto, caption="Vista previa", width=300)

    st.markdown("---")

    # Sección 6: Coordenadas
    st.markdown('<div class="sec">Sección 6 — Coordenadas UTM zona 19S (opcional)</div>',unsafe_allow_html=True)
    cx,cy,ci=st.columns([2,2,3])
    with cx:
        xv_str=st.text_input("X — Este:",placeholder="Ej: 315569.5",key="coord_x")
        try: xv=float(xv_str.replace(",",".")) if xv_str.strip() else 0.0
        except: xv=0.0
    with cy:
        yv_str=st.text_input("Y — Norte:",placeholder="Ej: 6136152.3",key="coord_y")
        try: yv=float(yv_str.replace(",",".")) if yv_str.strip() else 0.0
        except: yv=0.0
    with ci:
        if xv>1000 and yv>1000:
            lp,lop=u2l(xv,yv); st.success(f"Lat:{lp:.5f} | Lon:{lop:.5f}")
        else: st.info("Ingresa X e Y para identificar el alimentador")
    st.markdown("---")
    if st.button("Clasificar falla",type="primary"):
        errs=[]
        if not usuario.strip(): errs.append("Ingresa el nombre del operador.")
        if not desc.strip(): errs.append("Ingresa o selecciona una descripción.")
        for e in errs: st.error(e)
        if not errs:
            rf=predecir(desc)
            if cal_sug and rf["conf"]<60: rf["pred"]=cal_sug
            x_coord=float(xv) if xv>1000 else None
            y_coord=float(yv) if yv>1000 else None
            rg=id_alim(x_coord,y_coord)
            foto_bytes_val = foto.read() if foto else None
            foto_nombre_val = foto.name if foto else None
            st.session_state["ultimo"]={"usuario":usuario,"cuadrilla":cuadrilla,
                "desc":desc,"cat":cat,"sub":sub,"modo":modo,
                "rf":rf,"rg":rg,"x":x_coord,"y":y_coord,
                "trafos":trafos,"kva":kva,"clientes":clientes,
                "comuna":comuna_sel,"comuna_id":comuna_id,"tension":tension_sel,
                "inicio":dt_ini.strftime("%Y-%m-%d %H:%M"),
                "fin":dt_fin.strftime("%Y-%m-%d %H:%M"),
                "dur_hrs":round(dur_hrs,2),
                "foto_bytes":foto_bytes_val,
                "foto_nombre":foto_nombre_val}
            st.session_state["mostrar"]=True
    if st.session_state.get("mostrar") and st.session_state.get("ultimo"):
        u=st.session_state["ultimo"]; rf=u["rf"]; rg=u["rg"]
        x_coord=u["x"]; y_coord=u["y"]
        st.markdown(f"""<div class="resultado-{rf["pred"]}">
          <span class="badge-{rf["pred"]}">{ETIQUETAS[rf["pred"]]}</span>
          &nbsp;<span style="font-size:13px;color:#555">
            Confianza:{rf["conf"]}% | Operador:{u["usuario"]} | Cuadrilla:{u["cuadrilla"]}
          </span>
          <p style="margin:8px 0 0;font-size:13px"><b>Descripción:</b> {u["desc"]}</p>
        </div>""",unsafe_allow_html=True)
        p1,p2,p3=st.columns(3)
        p1.metric("FM",f"{rf[chr(70)+chr(77)]}%"); p1.progress(rf["FM"]/100)
        p2.metric("Externa",f"{rf[chr(69)]}%"); p2.progress(rf["E"]/100)
        p3.metric("Interna",f"{rf[chr(73)]}%"); p3.progress(rf["I"]/100)
        if rg:
            nivel="Alta" if rg["conf"]>=80 else "Media" if rg["conf"]>=50 else "Baja"
            votos=" | ".join(f"Alim.{a}:{p}%" for a,p in sorted(rg["votos"].items(),key=lambda x:-x[1]))
            st.markdown(f"""<div class="geo-card">
              <b>Alimentador identificado: #{rg["alim"]}</b> — Confianza {nivel} ({rg["conf"]}%)<br>
              Dist.punto histórico:{rg["dist_p"]}m | Comunas:{rg["comunas"]}<br>
              <small>Votos KNN: {votos}</small>
            </div>""",unsafe_allow_html=True)
            with st.expander("Ver mapa",expanded=True):
                m=folium.Map(location=[-34.98,-71.08],zoom_start=10)
                for alim,c in sorted(info_alim.items()):
                    col=COLORES_ALIM.get(alim,"gray"); es=rg["alim"]==alim
                    paso=max(1,len(c["pts"])//60)
                    for i in range(0,len(c["pts"]),paso):
                        folium.CircleMarker(c["pts"][i],radius=3,color=col,
                            fill=True,fill_opacity=0.35,weight=0).add_to(m)
                    folium.Marker([c["lat"],c["lon"]],tooltip=f"Alimentador {alim}",
                        popup=folium.Popup(f"<b>Alim.{alim}</b><br>Comunas:{c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}",max_width=180),
                        icon=folium.Icon(color="red" if es else col,
                            icon="bolt" if es else "info-sign",
                            prefix="fa" if es else "glyphicon")).add_to(m)
                if x_coord and y_coord:
                    lp,lop=u2l(x_coord,y_coord)
                    folium.Marker([lp,lop],tooltip="Punto de la falla",
                        popup=f"X:{x_coord:.0f} Y:{y_coord:.0f}",
                        icon=folium.Icon(color="black",icon="map-marker",prefix="fa")).add_to(m)
                    folium.PolyLine([[lp,lop],[info_alim[rg["alim"]]["lat"],info_alim[rg["alim"]]["lon"]]],
                        color="red",weight=3,dash_array="6").add_to(m)
                    m.location=[lp,lop]; m.zoom_start=13
                st_folium(m,width=700,height=400)
        else:
            st.info("Sin coordenadas — no se identificó el alimentador.")
        if u.get("foto_bytes"):
            st.markdown("**Foto adjunta:**")
            st.image(u["foto_bytes"], width=300)
        st.markdown("---")
        if st.button("Guardar registro",type="primary"):
            with st.spinner("Guardando en GitHub..."):
                n=guardar({
                    "Usuario":u["usuario"],"Cuadrilla":u["cuadrilla"],
                    "Descripcion":u["desc"],"Categoria":u["cat"],"Subcausa":u["sub"],
                    "Modo":"Guiado" if "Guiado" in u["modo"] else "Manual",
                    "Clasificacion":rf["pred"],
                    "Tipo_Falla":ETIQUETAS[rf["pred"]],
                    "Confianza_%":rf["conf"],"Prob_FM":rf["FM"],"Prob_E":rf["E"],"Prob_I":rf["I"],
                    "Trafos_afectados":u.get("trafos",""),
                    "KVA":u.get("kva",""),
                    "Clientes_afectados":u.get("clientes",""),
                    "Comuna":u.get("comuna",""),
                    "Comuna_ID":u.get("comuna_id",""),
                    "Tension":u.get("tension",""),
                    "Inicio":u.get("inicio",""),
                    "Fin":u.get("fin",""),
                    "Duracion_hrs":u.get("dur_hrs",""),
                    "X_UTM":x_coord or "","Y_UTM":y_coord or "",
                    "Alimentador":rg["alim"] if rg else "",
                    "Confianza_Alim_%":rg["conf"] if rg else "",
                    "Dist_punto_m":rg["dist_p"] if rg else "",
                    "Comunas_alim":rg["comunas"] if rg else ""},
                    foto_bytes=u.get("foto_bytes"),
                    foto_nombre=u.get("foto_nombre"))
            st.success(f"Registro N°{n} guardado correctamente en GitHub")
            st.session_state["mostrar"]=False
            st.session_state["ultimo"]={}

# ══════════════════════════════════════════════════════════
# PÁGINA 2 — HISTORIAL
# ══════════════════════════════════════════════════════════
elif pagina=="Historial":
    st.markdown("## Historial de Fallas")
    with st.spinner("Cargando historial desde GitHub..."):
        dh=cargar_hist()
    if dh.empty:
        st.info("Sin registros aún.")
    else:
        c1,c2,c3,c4=st.columns(4); c1.metric("Total",len(dh))
        if "Clasificacion" in dh.columns:
            cnt=dh["Clasificacion"].value_counts()
            c2.metric("FM",cnt.get("FM",0))
            c3.metric("E",cnt.get("E",0))
            c4.metric("I",cnt.get("I",0))
        st.markdown("---")
        fa,fb=st.columns(2)
        with fa:
            if "Clasificacion" in dh.columns:
                ft=st.multiselect("Filtrar tipo:",["FM","E","I"],default=["FM","E","I"])
                dh=dh[dh["Clasificacion"].isin(ft)]
        with fb:
            if "Usuario" in dh.columns:
                us=["Todos"]+sorted(dh["Usuario"].dropna().unique().tolist())
                fu=st.selectbox("Filtrar usuario:",us)
                if fu!="Todos": dh=dh[dh["Usuario"]==fu]
        st.dataframe(dh,use_container_width=True,height=400)
        col_d1,col_d2=st.columns(2)
        with col_d1:
            csv=dh.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV",csv,"historial_fallas.csv","text/csv")
        with col_d2:
            excel_buf=__import__("io").BytesIO()
            dh.to_excel(excel_buf,index=False); excel_buf.seek(0)
            st.download_button("Descargar Excel",excel_buf,
                "historial_fallas.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════
# PÁGINA 3 — MAPA GENERAL
# ══════════════════════════════════════════════════════════
elif pagina=="Mapa General":
    st.markdown("## Mapa de Alimentadores CEC")
    ma,mb=st.columns([3,1])
    with mb:
        st.markdown("**Alimentadores:**")
        for alim,c in sorted(info_alim.items()):
            st.markdown(f"Alim.{alim} — {c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}")
    with ma:
        m=folium.Map(location=[-34.98,-71.08],zoom_start=10)
        for alim,c in sorted(info_alim.items()):
            col=COLORES_ALIM.get(alim,"gray")
            paso=max(1,len(c["pts"])//60)
            for i in range(0,len(c["pts"]),paso):
                folium.CircleMarker(c["pts"][i],radius=3,color=col,
                    fill=True,fill_opacity=0.35,weight=0).add_to(m)
            folium.Marker([c["lat"],c["lon"]],tooltip=f"Alimentador {alim}",
                popup=folium.Popup(
                    f"<b>Alim.{alim}</b><br>"
                    f"Comunas:{c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}<br>"
                    f"Fallas:{c[chr(110)+chr(95)+chr(102)+chr(97)+chr(108)+chr(108)+chr(97)+chr(115)]}",
                    max_width=180),
                icon=folium.Icon(color=col,icon="info-sign",prefix="glyphicon")).add_to(m)
        st_folium(m,width=700,height=500)

# ══════════════════════════════════════════════════════════
# PÁGINA 4 — AYUDA
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
# PÁGINA 5 — ESTADÍSTICAS HISTÓRICAS
# ══════════════════════════════════════════════════════════
elif pagina=="Datos Históricos":
    st.markdown("## Estadísticas Históricas de Fallas (2018–2025)")
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except:
        st.error("Instala plotly: pip install plotly"); st.stop()

    # ── Calcular todo desde la hoja BD ─────────────────────────────────────
    @st.cache_data
    def calcular_historico(path):
        xl2 = pd.read_excel(path, sheet_name="BD")
        raw = xl2.iloc[1:].copy()
        raw.columns=["_","INC","Alimentador","Trafos","KVA","Clientes","Inicio","Fin",
                     "Duracion","Causa","Calificacion","Comuna_id","Comuna","Tension",
                     "X","Y","Descripcion","Validacion","año","Falta_coord"]
        raw["Calificacion"]=raw["Calificacion"].str.strip().str.upper()
        raw["Inicio"]=pd.to_datetime(raw["Inicio"],errors="coerce")
        raw["Duracion"]=pd.to_numeric(raw["Duracion"],errors="coerce")
        raw["año"]=pd.to_numeric(raw["año"],errors="coerce").astype("Int64")
        raw["Alimentador"]=pd.to_numeric(raw["Alimentador"],errors="coerce")
        df=raw.dropna(subset=["Inicio","Calificacion","año"])
        df=df[df["Calificacion"].isin(["I","FM","E"])]
        return df

    with st.spinner("Calculando estadísticas desde BD..."):
        df_hist = calcular_historico(EXCEL_PATH)

    # ── Tabla resumen por año (calculada desde BD) ─────────────────────────
    resumen = df_hist.groupby(["año","Calificacion"]).agg(
        Fallas=("INC","count"),
        Horas_prom=("Duracion","mean")
    ).reset_index()

    fallas_piv = resumen.pivot(index="año",columns="Calificacion",values="Fallas").fillna(0).astype(int)
    horas_piv  = resumen.pivot(index="año",columns="Calificacion",values="Horas_prom").fillna(0).round(4)
    total_fallas = df_hist.groupby("año")["INC"].count()
    total_horas  = df_hist.groupby("año")["Duracion"].mean().round(4)

    tabla = pd.DataFrame({
        "Año"          : fallas_piv.index,
        "Fallas E"     : fallas_piv.get("E",0).values,
        "Hrs prom E"   : horas_piv.get("E",0).values,
        "Fallas FM"    : fallas_piv.get("FM",0).values,
        "Hrs prom FM"  : horas_piv.get("FM",0).values,
        "Fallas I"     : fallas_piv.get("I",0).values,
        "Hrs prom I"   : horas_piv.get("I",0).values,
        "Total Fallas" : total_fallas.values,
        "Hrs prom Total": total_horas.values,
    })

    # ── Sección 1: Cantidad de fallas por año ─────────────────────────────
    st.markdown('<div class="sec">Cantidad de Fallas por Año y Tipo (2018–2025)</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        cant_m=resumen.copy()
        cant_m["Calificacion"]=cant_m["Calificacion"].map({"E":"Externa","FM":"Fuerza Mayor","I":"Interna"})
        fig1=px.bar(cant_m,x="año",y="Fallas",color="Calificacion",barmode="stack",
                    color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                    title="Fallas por año y tipo",
                    labels={"año":"Año","Fallas":"N° fallas","Calificacion":"Tipo"})
        fig1.update_layout(plot_bgcolor="white",xaxis=dict(tickmode="linear",dtick=1))
        st.plotly_chart(fig1,use_container_width=True)
    with col2:
        fig_pie=px.pie(
            values=[int(fallas_piv.get("FM",pd.Series([0])).sum()),
                    int(fallas_piv.get("E",pd.Series([0])).sum()),
                    int(fallas_piv.get("I",pd.Series([0])).sum())],
            names=["Fuerza Mayor","Externa","Interna"],
            color_discrete_sequence=["#e74c3c","#f39c12","#27ae60"],
            title="Proporción total 2018–2025")
        st.plotly_chart(fig_pie,use_container_width=True)

    st.markdown("---")

    # ── Sección 2: Horas promedio por año ─────────────────────────────────
    st.markdown('<div class="sec">Horas Promedio de Interrupción por Año (2018–2025)</div>',unsafe_allow_html=True)
    col3,col4=st.columns(2)
    with col3:
        hrs_m=resumen.copy()
        hrs_m["Calificacion"]=hrs_m["Calificacion"].map({"E":"Externa","FM":"Fuerza Mayor","I":"Interna"})
        fig2=px.bar(hrs_m,x="año",y="Horas_prom",color="Calificacion",barmode="group",
                    color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                    title="Horas promedio por tipo y año",
                    labels={"año":"Año","Horas_prom":"Hrs promedio","Calificacion":"Tipo"})
        fig2.update_layout(plot_bgcolor="white",xaxis=dict(tickmode="linear",dtick=1))
        st.plotly_chart(fig2,use_container_width=True)
    with col4:
        fig3=px.line(tabla,x="Año",y="Hrs prom Total",markers=True,
                     title="Horas promedio total por año",
                     labels={"Hrs prom Total":"Hrs promedio total"},
                     color_discrete_sequence=["#2c3e50"])
        fig3.update_layout(plot_bgcolor="white",xaxis=dict(tickmode="linear",dtick=1))
        st.plotly_chart(fig3,use_container_width=True)

    st.markdown("---")

    # ── Sección 3: Tabla completa calculada desde BD ───────────────────────
    st.markdown('<div class="sec">Tabla Completa 2018–2025 (calculada desde BD)</div>',unsafe_allow_html=True)
    st.dataframe(tabla,use_container_width=True,hide_index=True)
    csv_tabla=tabla.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar tabla CSV",csv_tabla,"historico_fallas.csv","text/csv")

    st.markdown("---")

    # ── Sección 4: Tasas de falla 2018-2022 (hoja TD) ─────────────────────
    st.markdown('<div class="sec">Tasa de Falla Promedio Anual 2018–2022 (hoja TD)</div>',unsafe_allow_html=True)
    xl_td = pd.read_excel(EXCEL_PATH, sheet_name="TD")
    td = xl_td.iloc[2:7].copy()
    td.columns = ["_","_2","Año","E","FM","I","_3","_4","_5","_6"]
    td = td[["Año","E","FM","I"]].dropna()
    td["Año"]=pd.to_numeric(td["Año"],errors="coerce").astype(int)
    for c in ["E","FM","I"]: td[c]=pd.to_numeric(td[c],errors="coerce")
    td_m=td.melt(id_vars="Año",value_vars=["E","FM","I"],var_name="Tipo",value_name="Tasa")
    td_m["Tipo"]=td_m["Tipo"].map({"E":"Externa","FM":"Fuerza Mayor","I":"Interna"})
    fig4=px.bar(td_m,x="Año",y="Tasa",color="Tipo",barmode="group",
                color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                title="Tasa de falla promedio anual 2018–2022",
                labels={"Tasa":"Tasa","Tipo":"Tipo de falla"})
    fig4.update_layout(plot_bgcolor="white",xaxis=dict(tickmode="linear",dtick=1))
    st.plotly_chart(fig4,use_container_width=True)
    st.caption("ℹ️ Las tasas de falla 2018–2022 provienen de la hoja TD del Excel original.")

    st.markdown("---")

    # ── Sección 5: Fallas por alimentador ─────────────────────────────────
    st.markdown('<div class="sec">Fallas por Alimentador (2018–2025)</div>',unsafe_allow_html=True)
    nombres_alim={2:"Alim.2 Morza",6:"Alim.6 Zapallar",7:"Alim.7 Los Niches",
                  8:"Alim.8 Industrial",9:"Alim.9 Los Queñes",
                  10:"Alim.10 La Laguna",11:"Alim.11 La Obra"}
    df_alim=df_hist.dropna(subset=["Alimentador"]).copy()
    df_alim["Alimentador"]=df_alim["Alimentador"].astype(int)
    df_alim["Nombre"]=df_alim["Alimentador"].map(nombres_alim).fillna(df_alim["Alimentador"].astype(str))

    alim_cant=df_alim.groupby(["Nombre","Calificacion"]).agg(
        Fallas=("INC","count"),
        Horas_prom=("Duracion","mean")
    ).reset_index()
    alim_cant["Calificacion"]=alim_cant["Calificacion"].map({"E":"Externa","FM":"Fuerza Mayor","I":"Interna"})

    col5,col6=st.columns(2)
    with col5:
        fig5=px.bar(alim_cant,x="Nombre",y="Fallas",color="Calificacion",barmode="stack",
                    color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                    title="Total fallas por alimentador")
        fig5.update_layout(plot_bgcolor="white",xaxis_tickangle=-20)
        st.plotly_chart(fig5,use_container_width=True)
    with col6:
        fig6=px.bar(alim_cant,x="Nombre",y="Horas_prom",color="Calificacion",barmode="group",
                    color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                    title="Horas promedio por alimentador y tipo")
        fig6.update_layout(plot_bgcolor="white",xaxis_tickangle=-20)
        st.plotly_chart(fig6,use_container_width=True)

    st.markdown("---")

    # ── Sección 6: Evolución mensual ───────────────────────────────────────
    st.markdown('<div class="sec">Evolución Mensual de Fallas (2018–2025)</div>',unsafe_allow_html=True)
    df_hist["Mes"]=df_hist["Inicio"].dt.to_period("M").astype(str)
    mensual=df_hist.groupby(["Mes","Calificacion"]).size().reset_index(name="Cantidad")
    mensual["Calificacion"]=mensual["Calificacion"].map({"E":"Externa","FM":"Fuerza Mayor","I":"Interna"})
    fig7=px.line(mensual,x="Mes",y="Cantidad",color="Calificacion",
                 color_discrete_map={"Fuerza Mayor":"#e74c3c","Externa":"#f39c12","Interna":"#27ae60"},
                 title="Evolución mensual 2018–2025",
                 labels={"Calificacion":"Tipo","Mes":"Mes","Cantidad":"N° fallas"})
    fig7.update_layout(plot_bgcolor="white",xaxis_tickangle=-45)
    st.plotly_chart(fig7,use_container_width=True)

# ══════════════════════════════════════════════════════════
# PÁGINA 6 — ESTADÍSTICAS ACTUALES
# ══════════════════════════════════════════════════════════
elif pagina=="Datos Actuales":
    st.markdown("## Fallas Registradas en la App")
    try:
        import plotly.express as px
    except:
        st.error("Instala plotly"); st.stop()
    with st.spinner("Cargando..."):
        dh=cargar_hist()
    if dh.empty or "Clasificacion" not in dh.columns:
        st.info("Sin registros aún. Registra algunas fallas primero.")
    else:
        dh["Fecha_hora"]=pd.to_datetime(dh["Fecha_hora"],errors="coerce")
        cnt=dh["Clasificacion"].value_counts()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total",len(dh))
        c2.metric("FM",cnt.get("FM",0))
        c3.metric("Externa",cnt.get("E",0))
        c4.metric("Interna",cnt.get("I",0))
        st.markdown("---")
        col1,col2=st.columns(2)
        with col1:
            fig1=px.pie(dh,names="Clasificacion",
                        color="Clasificacion",
                        color_discrete_map={"FM":"#e74c3c","E":"#f39c12","I":"#27ae60"},
                        title="Distribución de tipos de falla")
            st.plotly_chart(fig1,use_container_width=True)
        with col2:
            if "Cuadrilla" in dh.columns:
                qc=dh["Cuadrilla"].value_counts().reset_index()
                qc.columns=["Cuadrilla","N"]
                fig2=px.bar(qc,x="Cuadrilla",y="N",title="Fallas por cuadrilla",
                            color_discrete_sequence=["#3498db"])
                fig2.update_layout(plot_bgcolor="white",xaxis_tickangle=-30)
                st.plotly_chart(fig2,use_container_width=True)
        if "Alimentador" in dh.columns:
            st.markdown("---")
            dh2=dh[dh["Alimentador"].astype(str)!=""]
            if len(dh2)>0:
                ac=dh2.groupby(["Alimentador","Clasificacion"]).size().reset_index(name="N")
                fig3=px.bar(ac,x="Alimentador",y="N",color="Clasificacion",barmode="stack",
                            color_discrete_map={"FM":"#e74c3c","E":"#f39c12","I":"#27ae60"},
                            title="Fallas por alimentador")
                fig3.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig3,use_container_width=True)
        if "Duracion_hrs" in dh.columns:
            st.markdown("---")
            dh["Duracion_hrs"]=pd.to_numeric(dh["Duracion_hrs"],errors="coerce")
            dur_tipo=dh.groupby("Clasificacion")["Duracion_hrs"].mean().reset_index()
            dur_tipo.columns=["Tipo","Duración promedio (hrs)"]
            fig4=px.bar(dur_tipo,x="Tipo",y="Duración promedio (hrs)",
                        color="Tipo",
                        color_discrete_map={"FM":"#e74c3c","E":"#f39c12","I":"#27ae60"},
                        title="Duración promedio por tipo de falla")
            fig4.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig4,use_container_width=True)

        st.markdown("---")
        # Exportar PDF básico
        st.markdown("**Exportar resumen:**")
        csv=dh.to_csv(index=False).encode("utf-8")
        buf=__import__("io").BytesIO()
        dh.to_excel(buf,index=False); buf.seek(0)
        cd1,cd2=st.columns(2)
        with cd1:
            st.download_button("Descargar CSV",csv,"datos_actuales.csv","text/csv")
        with cd2:
            st.download_button("Descargar Excel",buf,"datos_actuales.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════
# PÁGINA 7 — MAPA DE CALOR
# ══════════════════════════════════════════════════════════
elif pagina=="Mapa de Calor":
    st.markdown("## Mapa de Calor de Fallas")
    from folium.plugins import HeatMap

    xl3=pd.read_excel(EXCEL_PATH,sheet_name=None)
    raw3=xl3["BD"].iloc[1:].copy()
    raw3.columns=["_","INC","Alimentador","Trafos","KVA","Clientes","Inicio","Fin",
                  "Duracion","Causa","Calificacion","Comuna_id","Comuna","Tension",
                  "X","Y","Descripcion","Validacion","año","Falta_coord"]
    def tf2(v):
        try: return float(str(v).replace(",","."))
        except: return np.nan
    raw3["X_f"]=raw3["X"].apply(tf2)
    raw3["Y_f"]=raw3["Y"].apply(tf2)
    raw3["Calificacion"]=raw3["Calificacion"].str.strip().str.upper()
    raw3["Duracion"]=pd.to_numeric(raw3["Duracion"],errors="coerce")
    raw3["Clientes"]=pd.to_numeric(raw3["Clientes"],errors="coerce")
    raw3.loc[(raw3["X_f"]<280000)|(raw3["X_f"]>380000),"X_f"]=np.nan
    raw3.loc[(raw3["Y_f"]<6080000)|(raw3["Y_f"]>6160000),"Y_f"]=np.nan
    df_hm=raw3.dropna(subset=["X_f","Y_f","Calificacion"]).copy()
    df_hm=df_hm[df_hm["Calificacion"].isin(["I","FM","E"])]
    df_hm["año"]=pd.to_numeric(df_hm["año"],errors="coerce")

    # Filtros
    st.markdown("### Filtros")
    fc1,fc2,fc3=st.columns(3)
    with fc1:
        tipos=st.multiselect("Tipo de falla:",["FM","E","I"],default=["FM","E","I"])
    with fc2:
        alims=sorted(df_hm["Alimentador"].dropna().astype(int).unique().tolist())
        alim_sel=st.multiselect("Alimentador:",alims,default=alims)
    with fc3:
        años=sorted(df_hm["año"].dropna().astype(int).unique().tolist())
        yr=st.select_slider("Rango de años:",options=años,value=(min(años),max(años)))
    fa1,fa2=st.columns(2)
    with fa1: radio_px=st.slider("Radio de puntos:",5,25,15)
    with fa2: blur_val=st.slider("Difuminado:",5,20,10)

    df_f=df_hm[
        (df_hm["Calificacion"].isin(tipos)) &
        (df_hm["Alimentador"].astype(float).astype(int).isin(alim_sel)) &
        (df_hm["año"].between(yr[0],yr[1]))
    ].copy()

    dur_max=df_f["Duracion"].max() if len(df_f)>0 and df_f["Duracion"].max()>0 else 1
    cli_max=df_f["Clientes"].max() if len(df_f)>0 and df_f["Clientes"].max()>0 else 1
    df_f["dur_norm"]=(df_f["Duracion"].fillna(0)/dur_max).clip(0.05,1)
    df_f["cli_norm"]=(df_f["Clientes"].fillna(0)/cli_max).clip(0.05,1)
    df_f["peso_total"]=(df_f["dur_norm"]+df_f["cli_norm"])/2

    st.markdown("---")
    st.markdown(f"**{len(df_f):,} fallas en el periodo {yr[0]}–{yr[1]}**")

    # Selector de mapa (radio en vez de tabs para compatibilidad con Streamlit Cloud)
    tipo_mapa=st.radio(
        "Seleccionar mapa:",
        ["Mapa 1 — Frecuencia de fallas",
         "Mapa 2 — Tiempo de desconexión",
         "Mapa 3 — Combinado (frecuencia + horas + clientes)"],
        horizontal=True,
        key="selector_mapa"
    )
    st.markdown("---")

    if len(df_f)==0:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        # Pre-calcular coordenadas una sola vez
        coords=[]
        for _,r in df_f.iterrows():
            la,lo=u2l(r["X_f"],r["Y_f"])
            coords.append((la,lo))

        if "Mapa 1" in tipo_mapa:
            st.markdown("**Zonas más intensas = mayor cantidad de fallas registradas**")
            m=folium.Map(location=[-34.98,-71.08],zoom_start=11,tiles="CartoDB positron")
            pts=[[la,lo] for la,lo in coords]
            HeatMap(pts,radius=radio_px,blur=blur_val,min_opacity=0.3).add_to(m)
            st_folium(m,width=900,height=500,key="hm1")
            st.markdown("**Distribución por tipo:**")
            c1,c2,c3=st.columns(3)
            for tipo,col in [("FM",c1),("E",c2),("I",c3)]:
                n=len(df_f[df_f["Calificacion"]==tipo])
                col.metric(tipo,f"{n:,}",f"{n/len(df_f)*100:.1f}%")

        elif "Mapa 2" in tipo_mapa:
            st.markdown("**Zonas más intensas = mayor tiempo acumulado de interrupción (horas)**")
            m=folium.Map(location=[-34.98,-71.08],zoom_start=11,tiles="CartoDB positron")
            pts=[[la,lo,float(df_f.iloc[i]["dur_norm"])] for i,(la,lo) in enumerate(coords)]
            HeatMap(pts,radius=radio_px,blur=blur_val,min_opacity=0.3).add_to(m)
            st_folium(m,width=900,height=500,key="hm2")
            d1,d2,d3=st.columns(3)
            d1.metric("Duración promedio",f"{df_f['Duracion'].mean():.2f} hrs")
            d2.metric("Duración máxima",f"{df_f['Duracion'].max():.2f} hrs")
            d3.metric("Total acumulado",f"{df_f['Duracion'].sum():.1f} hrs")
            st.markdown("**Top 5 fallas con mayor duración:**")
            top5=df_f.nlargest(5,"Duracion")[["año","Calificacion","Duracion","Clientes","Alimentador"]].copy()
            top5.columns=["Año","Tipo","Duración (hrs)","Clientes afectados","Alimentador"]
            st.dataframe(top5,use_container_width=True,hide_index=True)

        elif "Mapa 3" in tipo_mapa:
            st.markdown("**Zonas más intensas = mayor impacto combinado (frecuencia + horas + clientes afectados)**")
            m=folium.Map(location=[-34.98,-71.08],zoom_start=11,tiles="CartoDB positron")
            pts=[[la,lo,float(df_f.iloc[i]["peso_total"])] for i,(la,lo) in enumerate(coords)]
            HeatMap(pts,radius=radio_px,blur=blur_val,min_opacity=0.3).add_to(m)
            st_folium(m,width=900,height=500,key="hm3")
            e1,e2,e3,e4=st.columns(4)
            e1.metric("Total fallas",f"{len(df_f):,}")
            e2.metric("Horas acumuladas",f"{df_f['Duracion'].sum():.1f}")
            e3.metric("Clientes afectados",f"{int(df_f['Clientes'].sum()):,}")
            e4.metric("Promedio clientes/falla",f"{df_f['Clientes'].mean():.1f}")
            st.markdown("**Resumen por tipo de falla:**")
            res=df_f.groupby("Calificacion").agg(
                Fallas=("INC","count"),
                Hrs_prom=("Duracion","mean"),
                Hrs_total=("Duracion","sum"),
                Cli_prom=("Clientes","mean"),
            ).round(2).reset_index()
            res.columns=["Tipo","N° Fallas","Hrs promedio","Hrs totales","Clientes promedio"]
            st.dataframe(res,use_container_width=True,hide_index=True)


elif pagina=="Ayuda":
    st.markdown("## Guía de uso")
    with st.expander("¿Cómo registrar una falla?",expanded=True):
        st.markdown("""
        1. Ve a **Registrar Falla**
        2. Ingresa tu **nombre** y selecciona tu **cuadrilla**
        3. Elige **Guiado** (categoría + subcausa) o **Manual** (texto libre)
        4. Si tienes coordenadas X e Y, ingrésalas (acepta punto y coma)
        5. Presiona **Clasificar falla**
        6. Revisa el resultado y presiona **Guardar registro**
        """)
    with st.expander("Tipos de falla"):
        st.markdown("""
        | Tipo | Código | Descripción |
        |------|--------|-------------|
        | Fuerza Mayor | FM | Accidentes, vandalismo, catástrofes |
        | Externa | E | Fallas en generación o transmisión |
        | Interna | I | Fallas propias de la red CEC |
        """)
    with st.expander("Coordenadas y alimentadores"):
        st.markdown("""
        - Usa las coordenadas **X e Y** del Excel (UTM zona 19S)
        - Acepta tanto punto (315569.5) como coma (315569,5)
        - El modelo KNN identifica el alimentador con ~90% de precisión
        - Alta (>80%) / Media (50-80%) / Baja (<50%)
        """)
    with st.expander("¿Dónde se guardan los datos?"):
        st.markdown("""
        Los registros se guardan en **GitHub** en el archivo `fallas_ingresadas.csv`.
        Puedes descargarlo como **CSV o Excel** desde la página **Historial**.
        """)
