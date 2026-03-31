#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

const char* WIFI_SSID = "MECAP";
const char* WIFI_PASS = "8b140b20e7";

WebServer server(80);

// Ultrasonic pins
#define TRIG 14
#define ECHO 15

// Buzzer
#define BUZZER 13

long duration;
float distance;

bool yoloBeep = false;

// Camera resolutions
static auto loRes  = esp32cam::Resolution::find(320,240);
static auto midRes = esp32cam::Resolution::find(640,480);
static auto hiRes  = esp32cam::Resolution::find(800,600);


// ------------ CAMERA FUNCTIONS ------------

void serveJpg(){
  auto frame = esp32cam::capture();

  if(frame == nullptr){
    Serial.println("CAPTURE FAIL");
    server.send(503,"","");
    return;
  }

  server.setContentLength(frame->size());
  server.send(200,"image/jpeg");

  WiFiClient client = server.client();
  frame->writeTo(client);
}

void handleJpgLo(){
  esp32cam::Camera.changeResolution(loRes);
  serveJpg();
}

void handleJpgMid(){
  esp32cam::Camera.changeResolution(midRes);
  serveJpg();
}

void handleJpgHi(){
  esp32cam::Camera.changeResolution(hiRes);
  serveJpg();
}


// ------------ ULTRASONIC ------------

float getDistance(){

  digitalWrite(TRIG,LOW);
  delayMicroseconds(5);

  digitalWrite(TRIG,HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG,LOW);

  duration = pulseIn(ECHO,HIGH,30000);

  if(duration == 0){
    return -1;
  }

  distance = duration * 0.0343 / 2;

  return distance;
}


// ------------ DISTANCE API ------------

void handleDistance(){

  float d = getDistance();

  Serial.print("Distance: ");
  Serial.println(d);

  server.send(200,"text/plain",String(d));
}


// ------------ YOLO ALERT API ------------

void handleYolo(){

  Serial.println("YOLO detection received");

  yoloBeep = true;

  server.send(200,"text/plain","YOLO ALERT");
}


// ------------ SETUP ------------

void setup(){

  Serial.begin(115200);

  pinMode(TRIG,OUTPUT);
  pinMode(ECHO,INPUT);
  pinMode(BUZZER,OUTPUT);

  digitalWrite(BUZZER,LOW);

  {
    using namespace esp32cam;

    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(1);
    cfg.setJpeg(12);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID,WIFI_PASS);

  Serial.print("Connecting");

  while(WiFi.status()!=WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi Connected");

  Serial.print("http://");
  Serial.println(WiFi.localIP());

  Serial.println("/cam-lo.jpg");
  Serial.println("/cam-mid.jpg");
  Serial.println("/cam-hi.jpg");
  Serial.println("/distance");
  Serial.println("/yolo");

  server.on("/cam-lo.jpg",handleJpgLo);
  server.on("/cam-mid.jpg",handleJpgMid);
  server.on("/cam-hi.jpg",handleJpgHi);

  server.on("/distance",handleDistance);
  server.on("/yolo",handleYolo);

  server.begin();
}


// ------------ LOOP ------------

void loop(){

  server.handleClient();

  float d = getDistance();

  // continuous beep if object < 10cm
  if(d > 0 && d < 10){
    digitalWrite(BUZZER,HIGH);
  }
  else{
    digitalWrite(BUZZER,LOW);
  }

  // single beep for YOLO detection
  if(yoloBeep){
    digitalWrite(BUZZER,HIGH);
    delay(200);
    digitalWrite(BUZZER,LOW);
    yoloBeep = false;
  }

  delay(50);
}

