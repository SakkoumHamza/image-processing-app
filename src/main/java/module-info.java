module com.example.imageprocessingapp {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;
    requires commons.math3;
    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires net.synedra.validatorfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
    requires eu.hansolo.tilesfx;
    requires com.almasb.fxgl.all;
    requires opencv;
    requires javafx.swing;

    opens com.example.imageprocessingapp to javafx.fxml;
    exports com.example.imageprocessingapp;
}