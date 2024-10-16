import { Request } from 'express';
import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  Post,
  Query,
  Req,
  UnauthorizedException,
  UseGuards,
  ValidationPipe,
} from '@nestjs/common';

import { AppointmentsService } from './appointments.service';
import { ReportsService } from './reports/reports.service';
import { ScansService } from './scans/scans.service';
import { TreatmentService } from './treatments/treatments.service';
import { PrescriptionsService } from './prescriptions/prescriptions.service';
import { TestResultsService } from './test-results/test-results.service';
import { JwtGuard } from 'src/auth/guards/jwt.guard';

import {
  ApproveAppointmentDto,
  CompleteAppointmentDto,
  CreateAppointmentDto,
  GetAllAppointmentsQuery,
} from './dto/appointments.dto';
import type { Payload } from 'src/types/payload.type';

@Controller('appointments')
export class AppointmentsController {
  constructor(
    private readonly appointmentsService: AppointmentsService,
    private readonly reportsService: ReportsService,
    private readonly scansService: ScansService,
    private readonly treatmentService: TreatmentService,
    private readonly prescriptionsService: PrescriptionsService,
    private readonly testResultsService: TestResultsService,
  ) {}

  @UseGuards(JwtGuard)
  @Get()
  async getAllAppointments(
    @Query(new ValidationPipe({ transform: true }))
    query: GetAllAppointmentsQuery,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.findAll(query, undefined, user.id);
  }

  @UseGuards(JwtGuard)
  @Get(':id')
  async getAppointmentById(@Param('id') id: string, @Req() request: Request) {
    const user: Payload = request['user'];

    const appointment = await this.appointmentsService.findById(id);

    if (user.role === 'patient' && appointment.patientId !== user.id) {
      throw new UnauthorizedException();
    }

    return appointment;
  }

  @UseGuards(JwtGuard)
  @Get(':id/reports')
  async getAppointmentReportsById(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    const reports = await this.reportsService.findAllByAppointmentId(id);

    if (user.role === 'patient') {
      return reports.filter(
        (report) => report.appointment.patientId === user.id,
      );
    }

    return reports;
  }

  @UseGuards(JwtGuard)
  @Get(':id/scans')
  async getAppointmentScansById(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    const scans = await this.scansService.findAllByAppointmentId(id);

    if (user.role === 'patient') {
      return scans.filter((scan) => scan.appointment.patientId === user.id);
    }

    return scans;
  }

  @UseGuards(JwtGuard)
  @Get(':id/treatments')
  async getAppointmentTreatmentsById(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    const treatments = await this.treatmentService.findAllByAppointmentId(id);

    if (user.role === 'patient') {
      return treatments.filter(
        (treatment) => treatment.appointment.patientId === user.id,
      );
    }

    return treatments;
  }

  @UseGuards(JwtGuard)
  @Get(':id/prescriptions')
  async getAppointmentPrescriptionsById(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    const treatments =
      await this.prescriptionsService.findAllByAppointmentId(id);

    if (user.role === 'patient') {
      return treatments.filter(
        (treatment) => treatment.appointment.patientId === user.id,
      );
    }

    return treatments;
  }

  @UseGuards(JwtGuard)
  @Get(':id/test-results')
  async getAppointmentTestResultsById(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    const testResults =
      await this.testResultsService.findAllByAppointmentId(id);

    if (user.role === 'patient') {
      return testResults.filter(
        (testResult) => testResult.appointment.patientId === user.id,
      );
    }

    return testResults;
  }

  @UseGuards(JwtGuard)
  @Post()
  async createAppointment(
    @Body(new ValidationPipe({ transform: true })) dto: CreateAppointmentDto,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.create(user.id, dto);
  }

  @UseGuards(JwtGuard)
  @Patch(':id/approve')
  async approveAppointment(
    @Param('id') id: string,
    @Body(new ValidationPipe()) dto: ApproveAppointmentDto,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'admin') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.approve(id, dto.doctorId, user.id);
  }

  @UseGuards(JwtGuard)
  @Patch(':id/reject')
  async rejectAppointment(@Param('id') id: string, @Req() request: Request) {
    const user: Payload = request['user'];

    if (user.role !== 'admin') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.reject(id, user.id);
  }

  @UseGuards(JwtGuard)
  @Patch(':id/complete')
  async completeAppointment(
    @Param('id') id: string,
    @Body(new ValidationPipe()) dto: CompleteAppointmentDto,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.complete(id, dto.billingStatus, user.id);
  }

  @UseGuards(JwtGuard)
  @Patch(':id/cancel')
  async cancelAppointment(@Param('id') id: string, @Req() request: Request) {
    const user: Payload = request['user'];

    const appointment = await this.appointmentsService.findById(id);
    if (
      user.role !== 'admin' &&
      user.role === 'patient' &&
      appointment.patientId !== user.id
    ) {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.cancel(id, user.id);
  }
}
